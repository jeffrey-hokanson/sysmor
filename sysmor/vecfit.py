import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import solve_triangular, lstsq, svd
from .ratfit import RationalFit
from .pffit import PartialFractionRationalFit as PoleResidueRationalFit

class VFRationalFit(PoleResidueRationalFit):
	""" Use vector fitting to build a rational approximation


	Parameters
	----------
	m: int
		Degree of numerator
	n: int 
		Degree of denominator
	maxiter: int
		Maximum number of iterations
	"""
	def __init__(self, m, n, maxiter = 100, verbose = False, 
		tol = 1e-10, normalize = 'monic', **kwargs):

		assert m + 1 >= n, "Vector fitting can only handle degree m >= n - 1"
		self.m = m
		self.n = n

		self.maxiter = maxiter
		self.verbose = verbose
		self.tol = tol
		self.field = 'complex'
		self.normalize = normalize
		
		# TODO: checkme
		if 'init' not in kwargs:
			kwargs['init'] = 'linearize' 

	def _fit(self, lam0, W = None):
		if lam0 is None:
			lam0 = self._generate_zhat(self.n - 1)
			# need to make sure lam0 is not at any z
			while min([np.min(np.abs(lam0_ - self.z)) for lam0_ in lam0]) < 1e-5:
				lam0 += 1e-5

		if W is None:
			W = lambda x: x			

		lam = np.copy(lam0)
		N = len(self.f)

		if self.m - self.n >= 0:	
			V = self.legendre_vandmat(self.m - self.n, self.z)
		else:
			V = np.zeros((N,0))

		for it in range(self.maxiter):
			# Cauchy matrix
			C = 1./(np.tile(self.z.reshape(N,1), (1,self.n)) -  np.tile(lam.reshape(1,-1), (N, 1)))
			# Build bases for numerator and denominator
			Psi = np.hstack([np.ones((N,1)), C])
			Phi = np.hstack([C, V]) 
		
			# Setup, and optionally weight the linear system for the update step
			A = np.hstack( [Phi, -(Psi.T*self.f).T])
			A = W(A)
			
			# Compute the smallest singular values to solve the system
			if self.normalize == 'svd':
				U, s, VH = svd(A, full_matrices = False, compute_uv = True, lapack_driver = 'gesvd')
				x = VH.conj().T[:,-1]
				A_cond = s[0]/s[-1]
			else:
				I = np.ones(A.shape[1], dtype = np.bool)
				I[self.m+1] = 0
				x = np.ones(self.m+self.n+2, dtype = np.complex)
				x[I] = lstsq(A[:,I], -A[:,~I])[0].flatten()
				A_cond = np.linalg.cond(A)

			# Compute the coefficients on the polyomial
			a = x[:self.m+1]
			b = x[self.m+1:]

			a /= b[0]
			b /= b[0]

			# compute termination criteria
			b_norm = np.linalg.norm(b[1:])
			
			if self.verbose:
				# Evaluate the polynomials
				p = np.dot(Phi, a)
				q = np.dot(Psi, b)
				mismatch = p/q - self.f
				mismatch = W(mismatch)
				res = np.linalg.norm(mismatch)
				q_norm = np.linalg.norm(q - 1)

				if it == 0:
					print('  iter   |   residual   |  norm(q-1) |   cond(A)  |  norm(b[1:]) ')
					print('---------|--------------|------------|------------|--------------')
				print('    %3d  |  %1.4e  |  %1.2e  |  %1.2e  |  %1.2e' % (
					it, res, q_norm, A_cond, b_norm))
	
			# Don't scale by condition number because affects conversion
			if b_norm < self.tol:
				break

			if it < self.maxiter - 1:
				# Update roots only if we are going to continue the iteration
				# This is the root finding approach that Gustavsen takes in Vector Fitting
				# See Gus06: eq. 5
				lam = np.linalg.eigvals(np.diag(lam) - np.outer(np.ones(self.n), b[1:]))

		# NB: A better approach would be to explicitly convert into pole/residue form
		# from the polynomial ratio p/q, resolving for  
		# Copy over the values needed for the pole-residue expansion
		self.lam = np.copy(lam)
		self.rho_c = np.copy(a)	
		self.rho = self.rho_c[0:len(lam)]	

if __name__ == '__main__':
	import scipy.io
	dat = scipy.io.loadmat('data/fig_local_minima_cdplayer.mat')
	z = dat['z'].flatten()
	h = dat['h'].flatten()
	vf = VFRationalFit(29,30, verbose = True, maxiter = 500, tol_bnorm = 1e-14, normalize = 'monic')
	vf.fit(z, h)
	print(np.linalg.norm(vf(z) - h)/np.linalg.norm(h))
		
