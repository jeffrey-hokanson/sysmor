# (c) Jeffrey M. Hokanson May 2018
from __future__ import division
import numpy as np
from numpy.polynomial.legendre import legvander, legroots
from lagrange import LagrangePolynomial
 
class RationalFit:
	"""An abstract base class for rational approximation algorithms

	Subclasses of this abstract base class construct least squares rational approximations in the sense

	.. math::
	
		\min_{r \in \mathcal{R}_{m,n}(\mathbb{F}) } \| \mathbf{W} [ f(\mathcal Z) - z(\mathcal Z) ]\|_2,
		\qquad 
		f(\mathcal Z) := \\begin{bmatrix} f(z_1) \\\\ \\vdots \\\\ f(z_N) \end{bmatrix},
		\quad
		r(\mathcal Z) := \\begin{bmatrix} r(z_1) \\\\ \\vdots \\\\ r(z_N) \end{bmatrix}

	where :math:`\mathcal{R}_{m,n}(\mathbb{F})` denotes the class of rational functions of degree
	:math:`m,n` over the field :math:`\mathbb{F}`

	.. math::

		\mathcal{R}_{m,n}(\mathbb{F}) := 
		\left\lbrace 
			p/q, \ p \in \mathcal{P}_m(\mathbb{F}), \ q \in \mathcal{P}_n(\mathbb{F})
		\\right\\rbrace 

	and :math:`\mathcal{P}_m(\mathbb{F})` denotes the polynomials of degree :math:`m` over the field :math:`\mathbb{F}`.


	Parameters
	----------
	m: int
		Degree of numerator in rational approximation
	n: int 
		Degree of denominator in rational approximation
	field: {'complex', 'real'}, optional 
		Field over which to find the rational approximation; defaults to complex
	stable: bool, optional
		If true, restrict class of rational functions to those with poles in the open left half plane; defaults to False
	"""
	def __init__(self, m, n, field = 'complex', stable = False):
		assert m >= 0, "degree m must be a nonnegative integer"
		assert n >= 0, "degree n must be a nonnegative integer"
		assert field in ['complex', 'real'], "field must be either 'real' or 'complex'"
		self.m = m
		self.n = n
		self.field = field
		self.stable = stable


	def __call__(self, z):
		"""Evaluate the rational approximation at the given point(s)
		"""
		raise NotImplementedError

	def _set_scaling(self):
		# Constants ocassionaly used in fitting
		if self.real:
			# If real, we force z to be inside a unit box
			radius = max([np.max(np.abs(self.z.real)), np.max(np.abs(self.z.imag))])
			self._max_real =  radius
			self._min_real = -radius
			self._max_imag =  radius	
			self._min_imag = -radius
		else:
			self._max_real = np.max(self.z.real)
			self._min_real = np.min(self.z.real)
			self._max_imag = np.max(self.z.imag)
			self._min_imag = np.min(self.z.imag)

	def _init(self, W):
		return None

	def fit(self, z, f, lam0 = None, W = None):
		""" Fit a rational function to measurements 

		Solve the least squares rational approximation problem given pairs of 
		:math:`z_j, f(z_j)` stored in vectors `z` and `f`.

		Parameters
		----------
		z: array-like
			Samples corresponding to the argument of h
		h: array-like
			Samples of h(z)
		lam0: array-like (optional)
			Starting estimate of the pole locations (not used by all algorithms)
		W: matrix-like or function (optional)
			Weighting matrix associated with the data pairs.  Either provide a function
			that applies the matrix to a vector/matrix or a standard matrix. Defaults
			to applying no weight, i.e., :math:`\mathbf{W} = \mathbf{I}`.
		"""
		z = np.array(z)
		h = np.array(z)
		assert len(z.shape) == 1, "z must be a vector"
		assert len(h.shape) == 1, "h must be a vector"
		assert z.shape == h.shape, "z and h must have the same dimensions"
	
		self.z = np.copy(z)
		self.h = np.array(h, dtype = np.complex)
		
		if W is not None:
			if isinstance(self.W, np.array):
				self.W = lambda x: np.dot(W, x)
			else:
				self.W = W
		else:
			self.W = lambda x: x
	
		self._set_scaling()

		lam0 = self._init(W)

		self._fit(lam0)

	def to_system(self):
		""" Convert the rational approximation to a system 
		"""
		raise NotImplementedError

	def _transform(self, z):
		"""Apply shift-and-scale transform to input coordinates
		This transform maps the larger of the real or imaginary coordinates to [-1,1] 
		"""
		max_real = self._max_real
		min_real = self._min_real
		max_imag = self._max_imag
		min_imag = self._min_imag
	
		if (max_real - min_real) >= (max_imag - min_imag):
			zt = 2*(z - min_real - 1j* (max_imag + min_imag)/2.)/(max_real - min_real) - 1.
		else:
			zt = -2j*(z - 1j*min_imag - (max_real + min_real)/2.)/(max_imag - min_imag) - 1.
		return zt

	def _inverse_transform(self, zt):
		max_real = self._max_real
		min_real = self._min_real
		max_imag = self._max_imag
		min_imag = self._min_imag
	
		if (max_real - min_real) >= (max_imag - min_imag):
			z = (zt + 1)*(max_real - min_real)/2. + min_real + 1j*(max_imag + min_imag)/2.
		else:
			z = (zt + 1)*(max_imag - min_imag)/(-2j) + 1j * min_imag + (max_real + min_real)/2.

		return z

	def _legendre_vandmat(self, degree, z = None):
		""" A scaled and shifted Legendre polynomial basis		
		"""
		if z is None:
			z = self.z
		zt = self._transform(z) 
		return legvander(zt, degree)
	
	def _legendre_roots(self, c):
		""" Compute the roots of the scaled and shifted Legendre polynomial
		"""
		lam = legroots(c)
		lam = self._inverse_transform(lam)		

		return lam

	def _lagrange_vandmat(self, zhat, z = None):
		if z is None:
			z = self.z
		z_t = self._transform(z)
		zhat_t = self._transform(zhat)
		p = LagrangePolynomial(zhat_t)
		return p.vandmat(z_t)

	def _lagrange_roots(self, zhat, c):
		zhat_t = self._transform(zhat)
		p = LagrangePolynomial(zhat_t, c)
		lam_t = p.roots()
		lam = self._inverse_transform(lam_t)
		return lam

	def _generate_zhat(self, m):
		""" Generate well-distributed nodes
		"""
	
		# Place nodes at Chebyshev points of 2nd kind (on -1,1)
		if m == 0:
			zhat_t = np.array([0])
		else:
			zhat_t = np.cos( (np.pi*np.arange(m+1))/ m) 
		zhat = self._inverse_transform(zhat_t)
		return zhat


class OptimizationRationalFit(RationalFit):
	""" Parent class for optimization based approaches for rational fitting

	This class defines two initialization heuristics
	"""
	def _init_aaa(self, W):
		""" Use the AAA Algorithm to initialize the poles
		"""
		aaa = AAARationalFit(self.n)
		aaa.fit(self.z, self.h)
		lam = aaa.poles()		
		return lam
	

	def _init_recursive(self, W):
		m_orig = self.m
		n_orig = self.n
		if self.real: 
			step = 2
			if n_orig % 2 == 0: 
				# Even number of poles
				n = 2
			else:	
				# Odd number of poles
				n = 1
		else:
			n = 1 
			step = 1

		m = (m_orig - n_orig) + n 
		#m = n - 1
		
		# Constants we will need later
		max_real = np.max(self.z.real)
		min_real = np.min(self.z.real)
		max_imag = np.max(self.z.imag)
		min_imag = np.min(self.z.imag)
		
		# For the denominator we will use a barycentric polynomial
		# with nodes at the corners of the real or imaginary range 
		# (whichever is largest)
		if (max_real - min_real) > (max_imag - min_imag):	
			nodes = [min_real, max_real]
			nodes += [ (max_real + min_real)/2]
			nodes = np.array(nodes) + 1j*(min_imag + max_imag)/2.
		else:
			nodes = [1j*min_imag, 1j*max_imag]
			nodes += [ 1j*(min_imag + max_imag)/2. ]
			nodes = np.array(nodes) + (min_real + max_real)/2.
	
		lam = None
		res = np.copy(self.h)
		while n <= n_orig:
			num_degree = min(step, n)+1
			q = LagrangePolynomial(nodes[:num_degree])
			Psi = q.vandmat(self.z)

			p = LagrangePolynomial(nodes[:num_degree])
			Phi = p.vandmat(self.z)

			# Setup the "linearized" solution
			A = np.hstack([Phi, -np.dot(np.diag(res), Psi) ])
			A = self.W(A)
		
			#print "cond A %5.5e, cond Phi %5.5e, cond Psi %5.5e" % ( 
			#	np.linalg.cond(A), np.linalg.cond(Phi), np.linalg.cond(Psi))
			U, s, VH = np.linalg.svd(A)
			ab = VH.conjugate().T[:,-1]
			
			# Compute the new roots to add
			b = ab[-num_degree:]
			q = LagrangePolynomial(nodes[:num_degree], b)
			lam_new = q.roots()
			#lam_new = self.legendre_roots(b)
			#print lam_new
			# Force the new roots to come in conjugate pairs
			if self.real:
				I = marriage_sort(lam_new, lam_new.conjugate())
				lam_new = 0.5*(lam_new + lam_new[I].conjugate()) 

			# Either use these roots as the initial values 
			if lam is None: lam = lam_new
			# Or append them to the current list of roots
			else: lam = np.hstack([self.lam, lam_new])
			
			#I = np.argsort(lam.imag)
			#print "poles", lam[I]
			#print "lam new", lam_new
			
			# Fit to the current size
			self.m = m
			self.n = n
			if self.n == n_orig:
				#print "stopping with (%d,%d)" % (self.m, self.n)
				break

			self._fit(lam)
			res = self.residual(self.lam)
			#print "fitting (%d, %d)" % (m,n), "residual", np.linalg.norm(res)
			m += step
			n += step

		self.m = m_orig
		self.n = n_orig
		return lam				
	
