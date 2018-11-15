from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular, cholesky, svdvals
from h2mor import H2MOR
from pffit import PartialFractionRationalFit
from cauchy import cauchy_ldl 

def subspace_angle_V_M(mu, lam, L = None, d = None, p = None):
	"""Compute the subspace angles between V and M

	Defining the subspaces:

	.. math::

		\mathcal{V}(\\boldsymbol{\mu}) &:=  \lbrace v_\mu \\rbrace_{\mu \in \\boldsymbol{\mu}} \\\\
		\mathcal{M}(\\boldsymbol{\lambda}) &:=  \lbrace v_{-\overline{\lambda}}, v_{-\overline{\lambda}} \\rbrace_{\lambda \in \\boldsymbol{\lambda}}

	this function returns the canonical subspace angles between :math:`\mathcal{V}(\\boldsymbol{\mu})` 
	and :math:`\mathcal{M}(\\boldsymbol{\lambda})`.

	Parameters
	----------
	mu: array-like (n,)
		Parameters of the subspace :math:`\mathcal{V}` where :math:`\mu_j` is in the right half plane
	lam: array-like (m,)
		Parameters of the subspace :math:`\mathcal{M}` where :math:`\lambda_j` is in the left half plane
 

	Returns
	-------
	phi: np.array (min(n,m))
		The canonical subspace angles in radians
	"""
	mu = np.atleast_1d(np.array(mu, dtype = np.complex))
	lam = np.atleast_1d(np.array(lam, dtype = np.complex))
	assert np.all(mu.real > 0), "mu must be in right half plane"
	assert np.all(lam.real < 0), "lam must be in left half plane"

	# Compute Cholesky factorization of the left matrix
	if (L is None) or (d is None) or (p is None):
		L, d, p = cauchy_ldl(mu)	

	n = len(mu)
	m = len(lam)

	# Right matrix specifying normalization for basis 
	Mhat11 = (np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-1) 
	Mhat12 = (np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-2) 
	Mhat21 = ((np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-2)).conj().T 
	Mhat22 = 2*(np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-3)
	
	Mhat =  np.vstack([np.hstack([Mhat11, Mhat12]), np.hstack([Mhat21, Mhat22])]) 


	# Compute central matrix
	A11 = (np.tile(mu.reshape(n, 1), (1,m)) - np.tile(lam.reshape(1,m), (n,1)))**(-1)
	A12 = (np.tile(mu.reshape(n, 1), (1,m)) - np.tile(lam.reshape(1,m), (n,1)))**(-2)
	A = np.hstack([ A11, A12])

	# Cholesky factorization on right hand side
	R = cholesky(Mhat, lower = False)
	ARinv = solve_triangular(R, A.conj().T, lower = False, trans = 'C').conj().T
	LinvARinv = np.diag(d**(-0.5)).dot(solve_triangular(L, ARinv[p], lower = True, trans = 'N'))
	sigma = svdvals(LinvARinv)
	
	# Check that ill-conditioning hasn't affected us too much
	#assert np.all(sigma< 1.2), "Too ill-conditioned"
	sigma[sigma > 1] = 1.
	phi = np.arccos(sigma)
	return phi


# TODO: Move this function somewhere else
def cholesky_inv(f, L, d, p):
	""" Evaluate the weighted 2-norm associated with Cholesky factorization

	Given a permuted Cholesky factorization of a matrix :math:`\mathbf{M}`

	.. math::
		
		\mathbf{M} = \mathbf{P} \mathbf{L} \mathbf{D} \mathbf{L}^* \mathbf{P}^\\top

	with lower triangular matrix :math:`\mathbf{L}`, 
	diagonal matrix :math:`\mathbf{D}`, 
	and a permutation matrix :math:`\mathbf{P}`,
	evaluate the weight associated with :math:`\mathbf{M}^{-1}`.
	Namely, given a vector :math:`\mathbf{f}`, 
	we note evaluating the norm is equivalent to
	
	.. math:: 

		\mathbf{f}^* \mathbf{M}^{-1} \mathbf{f} = \| \mathbf{D}^{-1/2} \mathbf{L}^{-1} \mathbf{P} \mathbf{f}\|_2^2

	Here we return the interior of the 2-norm on the right.

	"""
	Linvf = solve_triangular(L, f[p], lower = True, trans = 'N')
	
	#return d.reshape(-1,1)**(-0.5)*Linvf
	return np.diag(d**(-0.5)).dot(Linvf)

class ProjectedH2MOR(H2MOR):
	""" Projected H2-optimal Model Reduction


	Parameters
	----------
	rom_dim: int
		Dimension of reduced order model to construct
	real: bool (optional)
		If True, fit a real dynamical system; if False, fi a complex dynamical system

	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000, verbose = False):
		H2MOR.__init__(self, rom_dim, real = real)
		self.maxiter = maxiter
		self.verbose = verbose

	def _mu_init(self, H):
		raise NotImplementedError

	def _fit(self, H, mu0 = None):
		
		if mu0 is None:
			self._mu_init(H)

		mu = np.array(mu0, dtype = np.complex)
		lam = None		# Poles of the previous iterate
		Hr = None
		
		# Outer loop
		for it in range(self.maxiter):
			lam_old = lam
			Hr_old = Hr
			n = len(mu)

			# Pick the order of rational approximation
			if n < 2*self.rom_dim:
				if self.real: 
					rom_dim = 2*(n//4)
				else: 
					rom_dim = (n//2)
			else:
				rom_dim = self.rom_dim

			# Initialize two copies of the fitting routine for the two initializations we will use 
			if self.real:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True) 
			else:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True) 

			# Evaluate the transfer function, recycling data
			H_mu = self.eval_transfer(H, mu)
			H_mu = H_mu.reshape(n,)		
	
			# Compute the weight matrix
			L,d,p = cauchy_ldl(mu) 
			M = lambda x: cholesky_inv(x, L, d, p)

			# Find rational approximation (inner loop)
			# Default (AAA) initialization
			Hr1.fit(mu, H_mu, W = M)

			if (lam_old is not None) and len(lam_old) == Hr2.n:
				# Initialization based on previous poles
				Hr2.fit(mu, H_mu, W = M, lam0 = lam_old)
				# Set the reduced order model to be the smaller of the two
				if Hr2.residual_norm() < Hr1.residual_norm():
					Hr = Hr2
				else:
					Hr = Hr1
			else:
				Hr = Hr1

			lam, _ = Hr.pole_residue()	
			
			# Check termination conditions
			

			# Find the largest subspace angle 
			print "lam", lam
			max_angles = np.zeros(len(lam))
			for i in range(len(lam)):
				max_angles[i] = np.max(subspace_angle_V_M(mu, lam[i], L = L, d = d, p = p))

			# Append existing mu
			i = np.argmax(max_angles)
			mu_star = -lam[i].conj()
			mu = np.hstack([mu, mu_star])
			if (np.abs(mu_star.imag) > 0) and self.real:
				mu = np.hstack([mu, mu_star.conj()])
			print "mu_star", mu_star
