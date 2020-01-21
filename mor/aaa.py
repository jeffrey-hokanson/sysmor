# An implementation of the AAA Algorithm by Nakatsukasa, Sete, and Trefethen 
# 
# Implemented by Jeffrey M. Hokanson
from __future__ import division
import numpy as np
import scipy
from .lagrange import BarycentricPolynomial 
from .ratfit import RationalFit

class AAARationalFit(RationalFit):
	""" Construct a degree (m,m) rational approximation using the AAA algorithm

	The Adaptive Anderson Antoulas (AAA) algorithm is a modification of the Loewner framework
	that appeared in [NST18]_. This algorithm builds a rational approximation that interpolates 
	the data at :math:`m+1` points and approximates on the remainder in a suboptimal manner.
	This is a suboptimal convex relatation that unlike optimization based approaches does not
	converge to spurious local minimizers.  However, due to the limitations of its derivation, this technique only
	supports rational functions of degree :math:`(m,m)`, cannot incorporate a weight matrix,
	and only works over rational functions on the complex field. 

	
	Parameters
	----------
	m: int, optional
		Degree of rational approximation. If not provided, the degree is chosen to match the data to a small tolerance
	tol: float, optional
		The target least squares
	verbose: bool, optional
		If true, print debugging information


	Attributes
	----------
	Ihat: array-like, bool
		Binary array the size of :code:`z`.  If :code:`Ihat[j] == True`, then 
		:code:`z[j]` is a location where the rational function interpolates
		the data.


	References
	----------
	.. [NST18] The AAA Algorithm for Rational Approximation. 
		Yuji Nakatsukasa, Olivier Sete, and Llyod N. Trefethen., 
		SIAM J. Sci. Comput. 2018, Vol 40., No. 3, pp. A1494--A1522
	"""  

	def __init__(self, m = None, tol = None, verbose = False):
		self.m = m 
		self.tol = tol
		self.verbose = verbose
	
		if self.m is None and self.tol is None:
			self.tol = 1e-13

		if self.tol is not None and self.m is None:
			self.m = 100

		if self.m is not None and self.tol is None:
			self.tol = 0.

		self.field = 'complex'
		self.W = lambda x: x
	
	def _fit(self, lam0 = None):

		residual = np.copy(self.f)
		self.Ihat = np.zeros(len(self.f), dtype = np.bool)
		for k in range(self.m+1):
			# support points
			Inew = np.argmax(np.abs(residual))
			self.Ihat[Inew] = True
			
			# Build the Loewner matrix
			L = self._build_loewner()
			
			# Compute the smallest right singular vector of L
			U, s, VH = scipy.linalg.svd(L, full_matrices = False, compute_uv = True, overwrite_a = True)
			self.b = VH.conjugate().T[:,-1] 
			
			# Update the residual
			residual = self.f - self.__call__(self.z)	

			if self.verbose:
				print("AAA iter %3d; residual norm %5.5e; sigma_min %5.5e" % (k, np.linalg.norm(residual), s[-1]))
			if self.tol is not None and np.linalg.norm(residual, np.inf) < self.tol:
				break

			# Stop if we can no no futher with the data we have	
			if k >= len(self.f)//2:
				break

	def _build_loewner(self):
		""" Construct the Loewner matrix corresponding to the current iterate
		"""
		zhat = self.z[self.Ihat]
		zcheck = self.z[~self.Ihat]

		nrows = np.sum(~self.Ihat)
		ncols = np.sum(self.Ihat)

		# Build the Cauchy matrix
		C = 1./(np.tile(zcheck.reshape(-1,1), (1,ncols)) - np.tile(zhat.reshape(1,-1), (nrows,1)))
		# Apply scaling to make it the Loewner matrix for this problem
		L = (C.T * self.f[~self.Ihat]).T - C*self.f[self.Ihat]
		return L

	def _call(self, zeval):
		zhat = self.z[self.Ihat]
		hhat = self.f[self.Ihat]
		with np.errstate(divide='ignore',invalid='ignore'):
			num = np.sum([ h*b/(zeval - z) for h, b, z in zip(hhat, self.b, zhat)], axis = 0)
			denom = np.sum([ b/(zeval - z) for b, z in zip(self.b, zhat)], axis = 0)
			reval = num/denom
		
		# Now check cases where we had a divide by zero error
		for j in np.argwhere(~np.isfinite(reval)):
			I = np.argmin(np.abs(zeval[j] - zhat))
			reval[j] = hhat[I]
		
		return reval


	def _pole_residue(self):
		zhat = self.z[self.Ihat]
		w = np.ones(zhat.shape)
		q = BarycentricPolynomial(zhat, self.b, w)
		lam = q.roots(deflation = False)
		I = np.argsort(lam.imag)
		lam = lam[I]

		dz = 1e-5*np.exp(2j*np.pi*np.linspace(0,1,4, endpoint = False))
		rho = [ np.dot(self.__call__(lamj + dz), dz)/len(dz) for lamj in lam ]
		rho = np.array(rho)

		return lam, rho


	def cleanup(self):
		""" Remove numerical Froissart doublets
		"""
		lam, rho = self.pole_residue()
	
		Ismall = np.argwhere(np.abs(rho) < 1e-13)
		for i in Ismall:
			# Remove support points near poles with small residues
			j = np.argmin(np.abs(lam[i] - self.z[self.Ihat]))
			j = np.argmin(np.abs(self.z - self.z[self.Ihat][j]))
			self.Ihat[j] = False
		# Fit again
		# Build the Loewner matrix
		L = self._build_loewner()
		
		# Compute the smallest right singular vector of L
		U, s, VH = scipy.linalg.svd(L, full_matrices = False, compute_uv = True, overwrite_a = True)
		self.b = VH.conjugate().T[:,-1] 
		
