import abc
import numpy as np
from .lagrange import LagrangePolynomial
from .pgf import PGF
from warnings import catch_warnings


import scipy
try:
	from scipy.linalg import solve_lyapunov
except:
	from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
import scipy.linalg
import scipy.sparse.linalg

from scipy.sparse.linalg import eigs, spsolve, LinearOperator
from scipy.linalg import eig, expm, block_diag, lu_factor, lu_solve, eigvals
from scipy.sparse import eye as speye
from scipy.sparse import diags as spdiag
from scipy.sparse import block_diag as spblock_diag
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.optimize import minimize
from numpy import eye
from numpy.linalg import solve
from copy import deepcopy


#__all__ = ['LTISystem', 'ComboSystem', 'StateSpaceSystem', 'SparseStateSpaceSystem', 
#		'TransferSystem', 'PoleResidueSystem', 'ZeroSystem']


def _make_dense_matrix(A):
	try:
		return A.toarray()
	except:
		return np.array(A)


class LTISystem(abc.ABC):
	""" Abstract base class for linear-time invariant systems
	"""

	@abc.abstractmethod
	def transfer(self, z, der = False, left_tangent = None, right_tangent = None):
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
	def shape(self):
		return (self.output_dim, self.input_dim)

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
		assert L> 0, "L=%g must be a positive scalar" % L
	
		# Quadrature points
		z = (1.j*L) / np.tan(np.arange(1, n+1) * np.pi / (n+1))
		
		# Quadrature weights; see eq. (3.7) DGB15
		w = L/(2*(n+1)*np.sin( np.arange(1,n+1)*np.pi/(n+1))**2)
		
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
		norm2 *= 1./(2*np.pi)	
		# Take the square root to return the actual norm
		if norm2 >= 0:
			norm = float(np.sqrt(norm2))
			return norm
		else:
			return np.nan

class ComboSystem(LTISystem):
	def __init__(*sys):
		pass


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
		self.A = _make_dense_matrix(A)
		self.B = _make_dense_matrix(B)
		self.C = _make_dense_matrix(C)	
		self.E = np.eye(A.shape[0])
		
		if len(B.shape) == 1:
			self.B = self.B.reshape(-1, 1)
		if len(C.shape) == 1:
			self.C = self.C.reshape(1, -1)
	
		# Ensure we cannot overwrite this data	
		for X in [self.A, self.B, self.C, self.E]:
			X.flags.writeable = False


	def _resolvent_solve(self, z, x, mode = 'N'):
		r""" Solve a linear system associated with this system

		If `mode='N'`:

		.. math:: 
			(z \mathbf{E} - \mathbf{A})^{-1} \mathbf{x}

		If `mode='T'`:
		
		.. math:: 
			
			(z \mathbf{E} - \mathbf{A})^{-T} \mathbf{x}
		
		If `mode='H'`:
		
		.. math:: 
			
			(z \mathbf{E} - \mathbf{A})^{-*} \mathbf{x}


		"""
		mode = mode.upper()
		assert mode in ['N', 'T', 'H']

		if mode == 'N':
			return scipy.linalg.solve(z * self.E - self.A, x)
		elif mode == 'T':
			return scipy.linalg.solve(z * self.E.T - self.A.T, x)
		elif mode == 'H':
			return scipy.linalg.solve(z * self.E.T.conj() - self.A.T.conj(), x)
		

	
	def transfer(self, z, der = False, left_tangent = None, right_tangent = None):
		z = np.atleast_1d(z)
		n = len(z)

		if left_tangent is None:
			C = self.C
		else:
			if len(left_tangent.shape) == 1:
				left_tangent = left_tangent.reshape(1,-1)
			C = left_tangent @ self.C

		if right_tangent is None:
			B = self.B
		else:
			if len(right_tangent.shape) == 1:
				right_tangent = right_tangent.reshape(-1,1)
			B = self.B @ right_tangent

		Hz = np.zeros((n, C.shape[0], B.shape[1]), dtype = np.complex)
		if der:
			Hpz = np.zeros((n, C.shape[0], B.shape[1]), dtype = np.complex)

		if B.shape[1] <= C.shape[0]:
			for i in range(n):
				# H(z) = C @ inv(z*E - A) @ B
				#R = scipy.linalg.solve(z[i] * self.E - self.A, B)
				R = self._resolvent_solve(z[i], B)
				Hz[i] = C @ R
				if der:
					# H'(z) = -C @ inv(z*E - A) @ E @ inv(z*E - A) @ B
					Hpz[i] = - C @ self._resolvent_solve(z[i], self.E @ R) 
		else:
			for i in range(n):
				R = self._resolvent_solve(z[i].conj(), C.conj().T, mode = 'H')
				Hz[i] = R.conj().T @ B
				if der:
					R2 = self._resolvent_solve(z[i].conj(), (R.conj().T @ self.E).conj().T, mode = 'H')
					Hpz[i] = -R2.conj().T @ B
				
			
		if der:
			return Hz, Hpz
		else:
			return Hz


	def __getitem__(self, key):
		"""Extract a subsystem component-wise
		"""
		if isinstance(key, tuple):
			C = self.C[key[0]]
			B = self.B[:,key[1]]
		else:
			C = self.C[key]
			B = self.B

		if isinstance(self, (StateSpaceSystem, SparseStateSpaceSystem)):
			return type(self)(self.A, B, C)
		elif isinstance(self, [DescriptorSystem]):
			return type(self)(self.A, B, C, self.E)


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
		#TODO: Check this is valid for complex systems as well
		lim_zH = self.C @ self.E @ self.B 
		return [lim_zH, lim_zH]


	def norm(self):
		r""" Computes the H2 norm
		"""
		if self.spectral_abscissa() >= 0:
			return np.inf
		# Replace with code that exploits Q is rank-1 and sparse structure for A
		norm2 = 0
		for i in range(self.input_dim):
			for j in range(self.output_dim):
				Q = -np.outer(self.B[:,i], self.B[:,i].conjugate())
				with catch_warnings(record = True) as w:
					X = solve_lyapunov(self.A, Q)
					if any([isinstance(w_, RuntimeWarning) for w_ in w]):
						print(w)
						return np.nan
					norm2_term = np.dot(self.C[j,:], np.dot(X, self.C[j,:].conjugate().T))
				if norm2_term < 0 and np.abs(norm2_term) < 1e-14:
					norm2_term = 0
				norm2 += norm2_term
		return np.sqrt(norm2.real)


	def __add__(self, other):
		if self.input_dim != other.input_dim:
			raise ValueError("Input dimensions must be the same")
		if self.output_dim != other.output_dim:
			raise ValueError("Output dimensions must be the same")

		# By default for now, we convert things that are state-space systems to
		# dense systems for the combination
		if isinstance(other, SparseStateSpaceSystem):
			A = block_diag(self.A, other.A.todense() )
		elif isinstance(other, (StateSpaceSystem, ZeroSystem)):
			A = block_diag(self.A, other.A)
		else: 
			raise NotImplementedError("Don't know how to combine these systems")

		B = np.vstack([self.B, other.B])
		C = np.hstack([self.C, other.C])

		return StateSpaceSystem(A, B, C)

	def __sub__(self, other):
		if self.input_dim != other.input_dim:
			raise ValueError("Input dimensions must be the same")
		if self.output_dim != other.output_dim:
			raise ValueError("Output dimensions must be the same")
		
		if isinstance(other, SparseStateSpaceSystem):
			A = block_diag(self.A, other.A.todense() )
			B = np.vstack([self.B, other.B])
			C = np.hstack([self.C, -1*other.C])
		elif isinstance(other, (StateSpaceSystem,ZeroSystem)):
			A = block_diag(self.A, other.A)
			B = np.vstack([self.B, other.B])
			C = np.hstack([self.C, -1*other.C])
		else: 
			raise NotImplementedError("Don't know how to combine these systems")

		return StateSpaceSystem(A, B, C)

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

	def pole_residue(self):
		r""" Compute the poles and residues of this system
		"""
		lam, V = scipy.linalg.eig(self.A)
		#B = V.dot(self.B)
		#C = scipy.linalg.solve(V.T, self.C.T).T
		B = scipy.linalg.solve(V, self.B)
		C = self.C.dot(V)
		rho = np.array([np.outer(B[i,:], C[:,i]) for i in range(len(lam))])
		return lam, rho

	def poles(self, which = 'all', k =  1):
		r"""Return the poles of the system

		The eigenvalues of :math:`\mathbf{A}` are the poles of the transfer function :math:`H`.
		Here we compute the eigenvalues of this matrix using a similar interface to :code:`eigs`

		Parameters
		----------
		which: ['LR','all']
			Which eigenvalues to compute 

			* LR: largest real

		k : int
			Number of poles to return	
		"""
		assert which in ['all', 'LR'], "invalid value for 'which'"

		ew = scipy.linalg.eig(self.A, left=False, right=False)
		if which == 'LR':
			I = np.argsort(-ew.real)
			return ew[I[:k]]
		elif which == 'all':
			return ew

	def eig(self, which = 'all', k = 1, left = False, right = False):
		out = scipy.linalg.eig(self.A, left = left, right = right)

		if which != 'all':
			if left is False and right is False:
				ew = out
			else:
				ew = out[0]
			
			if which =='LR':
				I = np.argsort(-ew.real)
			if left is False and right is False:
				return ew[I[:k]]
			else:
				return [ew[I[:k]]] + [o[:,I[:k]] for o in out[1:]]
		else:
			return out


	def to_diagonal(self):
		ew, evR = self.eig(right = True)
		B = scipy.linalg.solve(evR, self.B)
		C = self.C @ evR
		return DiagonalStateSpaceSystem(ew, B, C) 	


	def spectral_abscissa(self):
		ew = self.poles(which = 'LR', k = 1)
		return ew.real		

class SparseStateSpaceSystem(StateSpaceSystem):
	def __init__(self, A, B, C):
		self.A = csr_matrix(A)
		self.B = _make_dense_matrix(B)
		self.C = _make_dense_matrix(C)
		self.E = scipy.sparse.eye(A.shape[0])

	def _resolvent_solve(self, z, x, mode = 'N'):
		x = np.atleast_2d(x)
		out = np.zeros((self.A.shape[0], x.shape[1]), dtype = np.complex)

		if mode == 'N':
			AA = z*self.E - self.A
		elif mode == 'T':
			AA = z*self.E.T - self.A.T
		elif mode == 'H':
			AA = z*self.E.T.conj() - self.A.T.conj()
		
		# Convert to CSC for access effiency		
		AA = AA.tocsc()
		# Use GMRES with ILU preconditioning
		ilu = scipy.sparse.linalg.spilu(AA)
		M = scipy.sparse.linalg.LinearOperator(AA.shape, ilu.solve)
		for i in range(x.shape[1]):
			out[:,i], info = scipy.sparse.linalg.gmres(AA, x[:,i], tol = 1e-10, atol = 1e-10, M = M)
			assert info == 0, "GMRES Failed"

		return out

	def to_dense(self):
		return StateSpaceSystem(self.A.todense(), self.B, self.C)

class DiagonalStateSpaceSystem(SparseStateSpaceSystem):
	def __init__(self, ew, B, C):
		self.ew = np.array(ew)
		self.B = _make_dense_matrix(B)
		self.C = _make_dense_matrix(C)
		self.E = scipy.sparse.eye(ew.shape[0])

	@property
	def A(self):
		return scipy.sparse.diags(self.ew)

	def _resolvent_solve(self, z, x, mode = 'N'):
		if mode == 'N' or mode == 'T':
			if len(x.shape) == 1:
				return np.multiply(1./(z - self.ew), x)
			elif len(x.shape) == 2:
				return np.multiply(1./(z - self.ew)[:,None], x)
		elif mode == 'H':	
			if len(x.shape) == 1:
				return np.multiply(1./(z - self.ew.conj()), x)
			elif len(x.shape) == 2:
				return np.multiply(1./(z - self.ew[:,None]).conj(), x)

	def poles(self):
		return self.ew

	def to_dense(self):
		return StateSpaceSystem(np.diag(self.ew), self.B, self.C)

	def to_diagonal(self):
		return self

class DescriptorSystem(StateSpaceSystem):
	def __init__(self, A, B, C, E):
		self.A = _make_dense_matrix(A)
		self.B = _make_dense_matrix(B)
		self.C = _make_dense_matrix(C)	
		self.E = _make_dense_matrix(E)
		
		if len(B.shape) == 1:
			self.B = self.B.reshape(-1, 1)
		if len(C.shape) == 1:
			self.C = self.C.reshape(1, -1)
		 
		# Ensure we cannot overwrite this data	
		for X in [self.A, self.B, self.C, self.E]:
			X.flags.writeable = False

	def poles(self, which = 'all', k = 1):
		ew = scipy.linalg.eig(self.A, self.E, left=False, right=False)
		if which == 'LR':
			I = np.argsort(-ew.real)
			return ew[I[:k]]
		elif which == 'all':
			return ew
	
	def eig(self, which = 'all', k = 1, left = False, right = False):
		out = scipy.linalg.eig(self.A, self.E, left = left, right = right)

		print('ew', out)
		if which == 'all':
			return out

		if left is False and right is False:
			ew = out
		else:
			ew = out[0]
		
		if which =='LR':
			I = np.argsort(-ew.real)
		if left is False and right is False:
			return ew[I[:k]]
		else:
			return [ew[I[:k]]] + [o[:,I[:k]] for o in out[1:]]

	def to_state_space(self):
		# Here we use the SVD rather than directly inverting E to maintain symmetry
		U, s, VH = scipy.linalg.svd(self.E)
		A = np.diag(1./np.sqrt(s)) @ U.conj().T @ self.A @ VH.conj().T @ np.diag(1./np.sqrt(s))
		B = np.diag(1./np.sqrt(s)) @ VH @ self.B
		C = self.C @ U @ np.diag(1./np.sqrt(s))
		return StateSpaceSystem(A, B, C) 
	
	def to_diagonal(self):
		return self.to_state_space().to_diagonal()		

	
class SparseDescriptorSystem(DescriptorSystem, SparseStateSpaceSystem):
	def __init__(self, A, B, C, E):
		pass

	def to_dense(self):
		return DescriptorSystem(self.A.todense(), self.B, self.C, self.E.todense())


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

	def transfer(self, z, der = False):
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

		if lim_zH is not None:
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
		else:
			return self._lim_zH

	def transfer(self, z, der = False):
		n = len(z)
		if self._vectorized:
			Hz = self._scaling*(self._H(z)).reshape(n, self.output_dim, self.input_dim)
			if der:
				Hpz = self._scaling*(self._Hder(z)).reshape(n, self.output_dim, self.input_dim)
		else:
			Hz = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)
			for i in range(len(z)):
				Hz[i] = self._scaling*(self._H(z[i]).reshape(self.output_dim, self.input_dim))
			
			if der:
				Hpz = np.zeros((len(z), self.output_dim, self.input_dim), dtype = np.complex)
				for i in range(len(z)):
					Hpz[i] = self._scaling*(self._Hder(z[i]).reshape(self.output_dim, self.input_dim))
		
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


	def __sub__(self, other):	
		transfer = lambda z: self.transfer(z) - other.transfer(z)
		transfer_der = lambda z: self.transfer(z) - other.transfer(z)
		isreal = self.isreal and other.isreal
		lim_zH = [self.lim_zH[0] - other.lim_zH[0], self.lim_zH[1] - other.lim_zH[1]]
		return TransferSystem(transfer, transfer_der = transfer_der, lim_zH  = lim_zH)

	def __rsub__(self, other):	
		transfer = lambda z: other.transfer(z) - self.transfer(z)
		transfer_der = lambda z: other.transfer(z) - self.transfer(z)
		isreal = self.isreal and other.isreal
		lim_zH = [other.lim_zH[0] - self.lim_zH[0], other.lim_zH[1] - self.lim_zH[1]]
		return TransferSystem(transfer, transfer_der = transfer_der, lim_zH  = lim_zH)
	
	def __add__(self, other):	
		transfer = lambda z: self.transfer(z) + other.transfer(z)
		transfer_der = lambda z: self.transfer(z) + other.transfer(z)
		isreal = self.isreal and other.isreal
		lim_zH = [self.lim_zH[0] + other.lim_zH[0], self.lim_zH[1] + other.lim_zH[1]]
		return TransferSystem(transfer, transfer_der = transfer_der, lim_zH  = lim_zH)
	
	def __radd__(self, other):	
		transfer = lambda z: self.transfer(z) + other.transfer(z)
		transfer_der = lambda z: self.transfer(z) + other.transfer(z)
		isreal = self.isreal and other.isreal
		lim_zH = [self.lim_zH[0] + other.lim_zH[0], self.lim_zH[1] + other.lim_zH[1]]
		return TransferSystem(transfer, transfer_der = transfer_der, lim_zH  = lim_zH)

	def __rsub__(self, other):	
		transfer = lambda z: other.transfer(z) - self.transfer(z)
		transfer_der = lambda z: other.transfer(z) - self.transfer(z)
		isreal = self.isreal and other.isreal
		lim_zH = [other.lim_zH[0] - self.lim_zH[0], other.lim_zH[1] - self.lim_zH[1]]
		return TransferSystem(transfer, transfer_der = transfer_der, lim_zH  = lim_zH)




#class SparseStateSpaceSystem(StateSpaceSystem):
#	def __init__(self, A, B, C, E = None):
#		self._A = csr_matrix(A)
#		try:
#			self._B = self._B.todense() 
#		except:
#			self._B = np.array(B)
#		try:
#			self._C = self._C.todense() 
#		except:
#			self._C = np.array(C)
#
#
#
#	def __add__(self, other):
#		if self.input_dim != other.input_dim:
#			raise ValueError("Input dimensions must be the same")
#		if self.output_dim != other.output_dim:
#			raise ValueError("Output dimensions must be the same")
#
#		if isinstance(other, SparseStateSpaceSystem):
#			A = spblock_diag([self.A, other.A])
#		elif isinstance(other, StateSpaceSystem):
#			A = block_diag(self.A.todense(), other.A)
#
#		B = np.vstack([self.B, other.B])
#		C = np.hstack([self.C, other.C])
#		
#		if isinstance(other, SparseStateSpaceSystem):
#			return SparseStateSpaceSystem(A, B, C)
#		else:
#			return StateSpaceSystem(A, B, C)
#
#
#	def __sub__(self, other):
#		if self.input_dim != other.input_dim:
#			raise ValueError("Input dimensions must be the same")
#		if self.output_dim != other.output_dim:
#			raise ValueError("Output dimensions must be the same")
#
#		if isinstance(other, SparseStateSpaceSystem):
#			A = spblock_diag([self.A, other.A])
#		elif isinstance(other, StateSpaceSystem):
#			A = block_diag(self.A.todense(), other.A)
#
#		B = np.vstack([self.B, other.B])
#		C = np.hstack([self.C, -1*other.C])
#		
#		if isinstance(other, SparseStateSpaceSystem):
#			return SparseStateSpaceSystem(A, B, C)
#		else:
#			return StateSpaceSystem(A, B, C)
#	# Implement poles	
#	def pole_residue(self):
#		r""" Compute the poles and residues of this system
#		"""
#		# TODO Improve efficiency
#		lam, V = scipy.linalg.eig(self.A.toarray())
#		#B = V.dot(self.B)
#		#C = scipy.linalg.solve(V.T, self.C.T).T
#		B = scipy.linalg.solve(V, self.B)
#		C = self.C.dot(V)
#		rho = np.array([np.outer(B[i,:], C[:,i]) for i in range(len(lam))])
#		return lam, rho
#
#	def norm(self):
#		A = self.A.todense()
#		sys = StateSpaceSystem(A, self.B, self.C)
#		return sys.norm()	



class PoleResidueSystem(SparseStateSpaceSystem):
	def __init__(self, poles, residues):
		self._poles = np.atleast_1d(np.array(poles))
		n = len(self._poles)
		# TODO: What about MIMO pole residue systems
		self._residues = np.array(residues).reshape(n, 1,1)

	def transfer(self, z, der = False):
		if der:
			raise NotImplementedError

		n = len(z)
		Hz = np.zeros((n,self.output_dim, self.input_dim), dtype = np.complex)
		for lam, rho in zip(self._poles, self._residues):
			Hz += 	np.einsum('i, jk->ijk', 1./(z-lam), rho)
		return Hz			

	@property
	def A(self):
		return spdiag(self._poles)

	@property
	def B(self):
		return self._residues.reshape(-1,1)

	@property
	def C(self):
		return np.ones((1, len(self._poles)))

	@property
	def E(self):
		return spdiag(np.ones(len(self._poles))) 

	def poles(self):
		return self._poles

	@property
	def spectral_abscissa(self):
		return np.max(self._poles.real)


class ZeroSystem(StateSpaceSystem):
	def __init__(self, output_dim, input_dim):
		self._A = np.zeros((0, 0))
		self._B = np.zeros((0, input_dim))
		self._C = np.zeros((output_dim, 0))



if __name__ == '__main__':
	from demos import build_cdplayer

	model = build_cdplayer()
	print("H2 Norm = %1.2e" % model.norm())
	print((model - model).norm())
