from __future__ import division

import numpy as np
from lagrange import LagrangePolynomial
from pgf import PGF

try:
	from scipy.linalg import solve_lyapunov
except:
	from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov

from scipy.sparse.linalg import eigs, spsolve, LinearOperator
from scipy.linalg import eig, expm, block_diag, lu_factor, lu_solve, eigvals
from scipy.sparse import eye as speye
from scipy.sparse import block_diag as spblock_diag
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.optimize import minimize
from numpy import eye
from numpy.linalg import solve
from copy import deepcopy

import matplotlib.pyplot as plt

__all__ = ['LTISystem', 'ComboSystem', 'StateSpaceSystem', 'SparseStateSpaceSystem', 'TransferSystem', 'EmptySystem', 'PoleResidueSystem']


class LTISystem(object):
	""" Abstract base class for linear-time invariant systems
	"""
	def transfer(self, z, der = False):
		r"""Evaluate the transfer function of the system
	
		A dynamical system is uniquely defined in terms of its transfer function 
		:math:`H:\mathbb{C}\to \mathbb{C}` that is analytic in the open right half plane.

		Parameters
		----------
		z: array-like (n,)
			Points at which to evaluate the transfer function
		der: bool
			If True, return the derivative of the transfer function as well

		Returns
		-------
		Hz: array-like (n,input_dim, output_dim)
			Samples of the transfer function
		dHz: array-like (n,input_dim, output_dim)
			Derivative of the transfer function at z;
			only returned if :code:`der` is True.
		"""
		z = np.atleast_1d(z)
		assert len(z.shape) == 1, "Too many dimensions in input z"
		return self._transfer(z, der = der)
	
#	def impulse(self, t):
#		"""Evaluate the impulse response of the system
#		"""
#		raise NotImplementedError

	def __add__(self, G):
		""" Add two systems

		Given this system :math:`H` and another :math:`G`,
		add these two systems together such that the result adds the transfer functions;
		i.e., form :math:`G+H`.
		"""
		raise NotImplementedError

	def __sub__(self, G):
		""" Subtract two systems

		Given this system :math:`H` and another :math:`G`,
		add these two systems together such that the result adds the transfer functions;
		i.e., form :math:`H - G`.
		"""
		raise NotImplementedError

	def __mul__(self, const):
		""" Scalar multiplication of a transfer function
		"""
		raise NotImplementedError

	@property
	def input_dim(self):
		"""Dimension of the input space
		"""
		raise NotImplementedError

	@property
	def output_dim(self):
		"""Dimension of the output space
		"""
		raise NotImplementedError

	@property
	def lim_zH(self):
		r""" The limit along the imaginary axis
	
		This provides the two limits:
	
		.. math::

			M_{\pm}[H] := \lim_{\omega\to \pm \infty} i\omega H(i\omega)

		Returns
		-------
		M: tuple, length 2
			:math:`M_-[H]` and :math:`M_+[H]`.
		"""
		raise NotImplementedError


	@property
	def isreal(self):
		r""" If true, the system has a real representation

		Real systems have the property that

		.. math::
		
			\overline{H(z)} = H(\overline{z}).

		This can half the cost of performing operations on H.

		Returns
		-------
		bool:
			True if the system has a real representation; false otherwise.
		"""
		return self._isreal


	def quad_norm(self, L = 1, n = 200):
		r"""Evaluate the H2-norm using a quadrature rule.
		
		Here we use Boyd/Clenshaw-Curtis quadrature rule following [DGB15]_
		to estimate the :math:`\mathcal{H}_2` norm of this system:

		.. math::
		
			\| H \|_{\mathcal{H}_2}^2 &\approx \frac{|M_+[H]|^2 + |M_-[H]|^2}{4L(n+1)} + \sum_{j=1}^n w_j \|H(z_j)\|_F^2 \\
			w_j &= \frac{ L}{2(n+1) \sin^2(j\pi/(n+1))}, \ z_j = i L \cot(j \pi/(n+1)), \ M_{\pm}[H] = \lim_{\omega\to\pm\infty} i\omega H(i\omega)

		

		Parameters
		----------
		L: float
			Scaling factor in the quadrature rule
		n: int
			Number of samples to use in the quadrature rule

		References
		----------
		.. [DGB15] Quadrature-based vector fitting for discretized H2 approximation.
			Z. Drmac, S. Gugercin, and C. Beattie. 
			SIAM J. Sci. Comput. 37 (2015) pp. 2738--2753

		"""
	
		# Quadrature points
		z = (1.j*L) / np.tan(np.arange(1, n+1) * np.pi / (n+1))
		
		# Quadrature weights; see eq. (3.7) DGB15
		w = 1./(2*(n+1)*np.sin( np.arange(1,n)*np.pi/(n+1))**2)
		
		# Sample the transfer function
		# TODO: Exploit real structure if present to reduce calls

		Hz = self.transfer(z)

		# Evaluate the norm on the interior
		Hz_norm2 = np.sum(np.abs(Hz)**2, axis = (1,2))

		# Evalute the sume
		norm2 = np.sum(Hz_norm2*w)

		# Add the limit points
		lim_zH1, lim_zH2 = self.lim_zH

		norm2 += (np.sum(np.abs(lim_zH1)**2) + np.sum(np.abs(lim_zH2)**2))/(4*L*(n+1))

		# Take the square root to return the actual norm
		norm = np.sqrt(norm2)
		return norm


class ComboSystem(LTISystem):
	r""" Represents a sum of subsystems

	Given a set of systems :math:`H_1,H_2,\ldots`,
	this class represents the sum :math:`H`:
	
	.. math::
	   
		H = \sum_{i} H_i.

	We do not need to worry about scalar constants
	as these can be combined into the individual systems.

	"""
	def __init__(self, *args):
		self.subsystems = args

	def _transfer(self, z, der = False):
		Hz = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)
		if der:
			Hzp = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)

		# Evaluate the transfer function on each system
		for sys in self.subsystems:
			if der:
				Hz1, Hzp1 = sys.transfer(z, True)
				Hz += Hz1
				Hzp += Hzp1	
			else:
				Hz1 = sys.transfer(z, False)
				Hz += Hz1

		if der:
			return Hz, Hzp
		else:
			return Hz

	def __mul__(self, const):
		return ComboSystem(*[const*sys for sys in self.subsystems])

	def __rmul__(self, const):
		return ComboSystem(*[const*sys for sys in self.subsystems])



class TransferSystem(LTISystem):
	r""" A system specified in terms of its transfer function.

	This class describes systems in terms of a provided function.


	Parameters
	----------
	transfer: callable
		Function taking a single complex argument and 
		evaluating the transfer function at that point,
		returning a matrix of size (output_dim, input_dim).
	transfer_der: callable, optional
		Function taking a single complex argument and 
		evaluating the transfer function at that point,
		returning a matrix of size (output_dim, input_dim).
	input_dim: int, default:1
		Number of inputs
	output_dim: int, default:1
		Number of outputs	
	isreal: bool, default: False
		If True, the system is real; otherwise it is complex.	
	vectorized: bool
		If True, evaluate transfer and transfer_der as 
	"""
	def __init__(self, transfer, transfer_der = None, input_dim = 1, output_dim = 1, isreal = False, lim_zH = None,
			vectorized = False):

		if self.lim_zH is not None:
			self._lim_zH = [np.array(lim_zH[0]).reshape(output_dim, input_dim), np.array(lim_zH[1]).reshape(output_dim, input_dim)]
		else:
			self._lim_zH = None
		self._H = transfer
		self._Hder = transfer_der
		self._scaling = complex(1.)
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._isreal = isreal
		self._vectorized = vectorized

	@property
	def input_dim(self):
		return self._input_dim

	@property
	def output_dim(self):
		return self._output_dim

	@property
	def lim_zH(self):
		if self._lim_zH is None:
			raise NotImplementedError
		return self._lim_zH

	def _transfer(self, z, der = False):
		n = len(z)
		if self._vectorized:
			Hz = self._scaling*(self._H(z)).reshape(n, self.output_dim, self.input_dim)
			if der:
				Hpz = self._scaling*(self._Hder(z)).reshape(n, self.output_dim, self.input_dim)
		else:
			Hz = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)
			for i in range(len(z)):
				Hz[i] = self._scaling*(self._H(z).reshape(self.output_dim, self.input_dim))
			
			if der:
				Hzp = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)
				for i in range(len(z)):
					Hpz[i] = self._scaling*(self._Hder(z).reshape(self.output_dim, self.input_dim))
		
		if der:	
			return Hz, Hpz
		else:
			return Hz


	# Scalar multiplication

	def __mul__(self, other):
		ret = deepcopy(self)
		ret._scaling *= other
		return ret
	
	def __rmul__(self, other):
		ret = deepcopy(self)
		ret._scaling *= other
		return ret

class StateSpaceSystem(LTISystem):
	r"""Represents a continuous-time system specified in state-space form

	Given matrices :math:`\mathbf{A}\in \mathbb{C}^{n\times n}`,
	:math:`\mathbf{B}\in \mathbb{C}^{n\times p}`,
	and :math:`\mathbf{C}\in \mathbb{C}^{q\times n}`,
	this class represents the dynamical system

	.. math::

		\mathbf{x}'(t) &= \mathbf{A}\mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) \quad \mathbf{x}(0) = \mathbf{0} \\
		\mathbf{y}(t) &= \mathbf{C} \mathbf{x}(t).


	Parameters
	----------
	A: array-like (n,n)
		System matrix
	B: array-like (n,p)
		Matrix mapping input to state
	C: array-like (q,n)
		Matrix mapping state to output	

	"""

	def __init__(self, A, B, C):
		self._A = np.array(A)
		self._B = np.array(B)
		self._C = np.array(C)
		

		if len(self.B_.shape) == 1:
			self._B = self.B_.reshape(-1, 1)
		if len(self.C_.shape) == 1:
			self._C = self.C_.reshape(1, -1)

	@property
	def A(self):
		return self._A

	@property
	def B(self):
		return self._B

	@property
	def C(self):
		return self._C

	@property
	def E(self):
		return self._E

	@property
	def state_dim(self):
		return self.A.shape[0]

	@property
	def input_dim(self):
		return self.B.shape[1]

	@property
	def output_dim(self):
		return self.C.shape[0]

	@property
	def lim_zH(self):
		if self.E is None:
			lim_zH = np.dot(self.C, self.B)
		else:
			lim_zH = np.dot(self.C, spsolve(self.E, self.B))
		return complex(lim_zH)

	def pole_box(self):
		"""Compute the box in which the poles of the system live
			using the numerical range
		"""

		real = [None, None]
		imag = [None, None]

		use_sparse = issparse(self.A) 
		if self.E is not None:
			use_sparse = use_sparse & issparse(self.E)
			
		if use_sparse:
			if self.E is not None:
				A_herm = LinearOperator(self.A.shape, 
					matvec = lambda x: 0.5*spsolve(self.E, self.A.dot(x)) + 0.5*self.A.T.conj().dot(spsolve(self.E.conj().T, x))
					)
				A_skew = LinearOperator(self.A.shape, 
					matvec = lambda x: 0.5j*spsolve(self.E, self.A.dot(x)) - 0.5j*self.A.T.conj().dot(spsolve(self.E.conj().T, x))
					)
			else:
				A_herm = 0.5 * (self.A + self.A.T.conjugate())
				A_skew = 0.5j * (self.A - self.A.T.conjugate())
			
			# Compute the largest real part of the hermitian part (right most limit)	
			ew, ev = eigs(A_herm, k=1, which='LR')
			real[1] = float(ew.real)
			# Compute the smallest real part of the hermitian part (left most limit)	
			ew, ev = eigs(A_herm, k=1, which='SR')
			real[0] = float(ew.real)

			# Compute the largest real part of the skew-hermitian part (top most limit)	
			ew, ev = eigs(A_skew, k=1, which='LR')
			imag[1] = float(ew.real)
			# Compute the smallest real part of the skew-hermitian part (bottom most limit)	
			ew, ev = eigs(A_skew, k=1, which='SR')
			imag[0] = float(ew.real)

		else:
			if self.E is not None:
				if issparse(self.A):
					A = self.A.todense()
				else:
					A = self.A
				if issparse(self.E):
					E = self.E.todense()
				else:
					E = self.E

				A = solve(E, A)
				A_herm = 0.5*(A + A.conj().T)
				A_skew = 0.5j*(A - A.conj().T)
			else:
				A_herm = 0.5*(self.A + self.A.conj().T)
				A_skew = 0.5j*(self.A - self.A.conj().T)
			
			ew = eig(A_herm, left=False, right=False)
			real[1] = np.max(ew.real)
			real[0] = np.min(ew.real)

			ew = eig(A_skew, left=False, right=False)
			imag[0] = np.min(ew.real)
			imag[1] = np.max(ew.real)

		return real, imag

	def norm(self):
		if self.E is not None:
			raise NotImplementedError

		if self.spectral_abscissa() >= 0:
			return np.inf
		# Replace with code that exploits Q is rank-1 and sparse structure for A
		Q = -np.outer(self.B, self.B.conjugate())
		if issparse(self.A):
			A = self.A.todense()
		else:
			A = self.A
		X = solve_lyapunov(A, Q)
		#np.savetxt('norm_%d_A.dat' % A.shape[0], A.view(float))
		#np.savetxt('norm_%d_B.dat' % A.shape[0], self.B.view(float))
		#np.savetxt('norm_%d_C.dat' % A.shape[0], self.C.view(float))
		#np.savetxt('norm_%d_X.dat' % A.shape[0], X.view(float))

		pre_norm = np.dot(self.C, np.dot(X, self.C.conjugate().T))
		norm = np.sqrt(np.dot(self.C, np.dot(X, self.C.conjugate().T)))
		return float(norm.real)

	def impulse(self, t):
		if self.E is not None:
			raise NotImplementedError

		output = np.dot(self.C, np.dot(expm(t * self.A), self.B))
		return output.reshape(self.output_dim, self.input_dim)

	def transfer(self, z):
		if issparse(self.A):
			# sparse version
			if self.E is None:
				E = csr_matrix(speye(self.state_dim))
			else:
				E = self.E
			EA = z * E - self.A
			x = spsolve(EA, self.B)
		else:
			# dense version
			if self.E is None:
				E = eye(self.state_dim)
			else:
				E = self.E
			x = solve(z * E - self.A, self.B)

		output = np.dot(self.C, x)
		return output.reshape(self.output_dim, self.input_dim)

	def transfer_der(self, z):
		#TODO: What is E is dense and A is sparse?
		if issparse(self.A):
			# sparse version
			if self.E is None:
				E = csr_matrix(speye(self.state_dim))
			else:
				E = self.E
			EA = z * E - self.A
			x = spsolve(EA, self.B)
			x_der = spsolve(EA, x)
		else:
			# dense version
			if self.E is None:
				E = eye(self.state_dim)
			else:
				E = self.E
			EA = z * E - self.A
			x = solve(EA, self.B)
			x_der = solve(EA, x)

		H_z = np.dot(self.C, x)
		Hp_z = np.dot(-self.C, x_der)
		return H_z.reshape(self.output_dim, self.input_dim), Hp_z.reshape(self.output_dim, self.input_dim)

	def __add__(self, other):
		if self.input_dim != other.input_dim:
			raise ValueError("Input dimensions must be the same")
		if self.output_dim != other.output_dim:
			raise ValueError("Output dimensions must be the same")

		# Combine A
		if issparse(self.A) and issparse(other.A):
			A = spblock_diag([self.A, other.A])
		elif not issparse(self.A) and issparse(other.A):
			A = block_diag(self.A, other.A.todense())
		elif issparse(self.A) and not issparse(other.A):
			A = block_diag(self.A.todense(), other.A)
		elif not issparse(self.A) and not issparse(other.A):
			A = block_diag(self.A, other.A)
		else:
			raise NotImplementedError('This should never be called')

		B = np.vstack([self.B, other.B])
		C = np.hstack([self.C, other.C])

		# Combine E
		if self.E is None and other.E is None:
			E = None
		elif self.E is None and issparse(other.E):
			E = spblock_diag([speye(self.state_dim), other.E])
		elif self.E is None and not issparse(other.E):
			E = block_diag(np.eye(self.state_dim), other.E)
		elif issparse(self.E) and other.E is None:
			E = spblock_diag([self.E, speye(other.state_dim)])
		elif not issparse(self.E) and other.E is None:
			E = block_diag(self.E, np.eye(other.state_dim))
		elif issparse(self.E) and issparse(other.E):
			E = spblock_diag([self.E, other.E])
		elif issparse(self.E) and not issparse(other.E):
			E = block_diag(self.E.todense(), E)
		elif not issparse(self.E) and issparse(other.E):
			E = block_diag(self.E, other.E.todense())
		elif not issparse(self.E) and not issparse(other.E):
			E = block_diag(self.E, other.E)
		else:
			raise NotImplementedError('This should never be called')

		return StateSpaceSystem(A, B, C, E = E)

	def __sub__(self, other):
		if self.input_dim != other.input_dim:
			raise ValueError("Input dimensions must be the same")
		if self.output_dim != other.output_dim:
			raise ValueError("Output dimensions must be the same")

		# Combine A
		if issparse(self.A) and issparse(other.A):
			A = spblock_diag([self.A, other.A])
		elif not issparse(self.A) and issparse(other.A):
			A = block_diag(self.A, other.A.todense())
		elif issparse(self.A) and not issparse(other.A):
			A = block_diag(self.A.todense(), other.A)
		elif not issparse(self.A) and not issparse(other.A):
			A = block_diag(self.A, other.A)
		else:
			raise NotImplementedError('This should never be called')

		B = np.vstack([self.B, other.B])
		C = np.hstack([self.C, -1.*other.C])

		# Combine E
		if self.E is None and other.E is None:
			E = None
		elif self.E is None and issparse(other.E):
			E = spblock_diag([speye(self.state_dim), other.E])
		elif self.E is None and not issparse(other.E):
			E = block_diag(np.eye(self.state_dim), other.E)
		elif issparse(self.E) and other.E is None:
			E = spblock_diag([self.E, speye(other.state_dim)])
		elif not issparse(self.E) and other.E is None:
			E = block_diag(self.E, np.eye(other.state_dim))
		elif issparse(self.E) and issparse(other.E):
			E = spblock_diag([self.E, other.E])
		elif issparse(self.E) and not issparse(other.E):
			E = block_diag(self.E.todense(), E)
		elif not issparse(self.E) and issparse(other.E):
			E = block_diag(self.E, other.E.todense())
		elif not issparse(self.E) and not issparse(other.E):
			E = block_diag(self.E, other.E)
		else:
			raise NotImplementedError('This should never be called')

		return StateSpaceSystem(A, B, C, E = E)

	def __mul__(self, other):
		ret = deepcopy(self)
		ret.C_ *= other
		return ret

	def __rmul__(self, other):
		ret = deepcopy(self)
		ret.C_ *= other
		return ret

	@property
	def isreal(self):
		if self.E is None:
			return np.isrealobj(self.A) & np.all(np.isreal(self.B)) & np.all(np.isreal(self.C))
		else:
			return np.isrealobj(self.A) & np.all(np.isreal(self.B)) & np.all(np.isreal(self.C)) & np.isrealobj(self.E)

	def poles(self):
		if self.E is not None:
			raise NotImplementedError
		if issparse(self.A):
			ew = eig(self.A.todense(), left=False, right=False)
		else:
			ew = eig(self.A, left=False, right=False)
		return ew

	def spectral_abscissa(self):
		if self.E is not None:
			raise NotImplementedError
		if issparse(self.A):
			ew = eigs(self.A, 1, which = 'LR', return_eigenvectors = False)
			return float(ew.real) 
		else:
			ew = eigvals(self.A)
			return np.max(ew.real)


class SparseStateSpaceSystem(StateSpaceSystem):
	def __init__(self, A, B, C, E = None):
		pass


class PoleResidueSystem(StateSpaceSystem):
	def __init__(self, poles, residues):
		self._poles = np.copy(poles)
		self._residues = np.copy(residues)
		A = np.diag(poles)
		C = np.copy(residues)
		B = np.ones(len(poles))

		# TODO: Option to create real valued A, B, C if poles and residues correspond to real system
		StateSpaceSystem.__init__(self, A, B, C)
		# TODO: Evaluate transfer function 



class EmptySystem(StateSpaceSystem):
	def __init__(self):
		self.A_ = np.zeros((0, 0))
		self.B_ = np.zeros((0, 1))
		self.C_ = np.zeros((1, 0))


if __name__ == '__main__':
	from test_cases import build_cdplayer

	model = build_cdplayer()
	print "H2 Norm = %1.2e" % model.norm()
	model.bode()
