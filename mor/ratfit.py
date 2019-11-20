# (c) Jeffrey M. Hokanson May 2018
from __future__ import division
import numpy as np
from numpy.polynomial.legendre import legvander, legroots
from .lagrange import LagrangePolynomial


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

		# Default weighting matrix: identity
		self.W = lambda x: x

		# set the scaling 
		self._max_real = 0.5
		self._min_real = -0.5
		self._min_imag = -0.5
		self._max_imag = 0.5

	def __call__(self, z):
		"""Evaluates the rational approximation

		Parameters
		----------
		z: array-like
			points to evaluate the rational approximation at
		
		Returns
		-------
		rz: array-like
			Evaluates this rational function
		"""
		return self._call(z)

	def _set_scaling(self):
		# Constants ocassionaly used in fitting
		if self.field == 'real':
			# If real, we force z to be inside a unit box
			#radius = max([np.max(np.abs(self.z.real)), np.max(np.abs(self.z.imag))])
			radius = np.max(np.abs(self.z.imag))
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
		f: array-like
			Samples of f(z)
		lam0: array-like (optional)
			Starting estimate of the pole locations (not used by all algorithms)
		W: matrix-like or function (optional)
			Weighting matrix associated with the data pairs.  Either provide a function
			that applies the matrix to a vector/matrix or a standard matrix. Defaults
			to applying no weight, i.e., :math:`\mathbf{W} = \mathbf{I}`.
		"""
		z = np.array(z)
		f = np.array(f)
		assert len(z.shape) == 1, "z must be a vector"
		assert len(f.shape) == 1, "h must be a vector"
		assert z.shape == f.shape, "z and h must have the same dimensions"
	
		self.z = np.copy(z)
		self.f = np.array(f, dtype = np.complex)
		
		if W is not None:
			if isinstance(W, np.ndarray):
				self.W = lambda x: np.dot(W, x)
			else:
				self.W = W
		else:
			self.W = lambda x: x
	
		self._set_scaling()
		if lam0 is None:
			lam0 = self._init(W)
		self._fit(lam0)


	def residual_norm(self):
		""" Computes the (weighted) 2-norm of the residual
		"""

		res = self.W(self.f - self.__call__(self.z))
		return np.linalg.norm(res, 2)


	def pole_residue(self):
		"""Return the poles and residues of the rational function

		"""
		return self._pole_residue()	
	

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
