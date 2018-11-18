from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular, cholesky, svdvals
from system import StateSpaceSystem, PoleResidueSystem, ZeroSystem
from h2mor import H2MOR
from pffit import PartialFractionRationalFit
from cauchy import cauchy_ldl, cauchy_hermitian_svd 
from marriage import marriage_sort

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
	phi: np.array (min(n,2*m))
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
	#delta = np.max(np.abs(lam))
	Mhat11 = (np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-1) 
	Mhat12 = (np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-2) 
	Mhat21 = ((np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-2)).conj().T 
	Mhat22 = 2*(np.tile(-lam.conj().reshape(m,1), (1,m)) - np.tile(lam.reshape(1,m), (m,1)))**(-3)
	
	Mhat =  np.vstack([np.hstack([Mhat11, Mhat12]), np.hstack([Mhat21, Mhat22])]) 
	
	# Compute central matrix
	A11 = (np.tile(mu.reshape(n, 1), (1,m)) - np.tile(lam.reshape(1,m), (n,1)))**(-1)
	A12 = (np.tile(mu.reshape(n, 1), (1,m)) - np.tile(lam.reshape(1,m), (n,1)))**(-2)
	A = np.hstack([ A11, A12])

	#R = cholesky(Mhat, lower = False)
	#print np.linalg.inv(R)
	#print solve_triangular(R, A.conj().T, lower = False, trans = 'C').conj().T
	if m > 1:
		# Cholesky factorization on right hand side
		# Mhat = R^* R
		R = cholesky(Mhat, lower = False)
		# A Mhat^{-1/2} = A R^{-1/2} = (R^{-*} A.H).H
		ARinv = solve_triangular(R, A.conj().T, lower = False, trans = 'C').conj().T
		# M^{-1/2} A Mhat^{-1/2} = L^{-1/2} A R^{-1/2}
	else:
		# Explicity Cholesky inverse of Mhat
		lam = lam[0].real
		Rinv = np.array([[np.sqrt(-2*lam), -np.sqrt(-2*lam)],[0, 2*np.sqrt(2)*(-lam)**(3/2)]])
		ARinv = A.dot(Rinv)
	LinvARinv = np.diag(d**(-0.5)).dot(solve_triangular(L, ARinv[p], lower = True, trans = 'N'))
	sigma = svdvals(LinvARinv)
	
	# Check that ill-conditioning hasn't affected us too much
	#assert np.all(sigma< 1.2), "Too ill-conditioned"
	sigma[sigma > 1] = 1.
	phi = np.arccos(sigma)
	#print "lam" , lam, np.max(phi)/np.pi*180., np.linalg.cond(Mhat)
	# Hackish solution for ill-conditioned Mhat matrices:
	#if np.linalg.cond(Mhat) > 1e6:
	#	phi = np.nan*phi 
	return phi


def subspace_angle_V_V(mu, hmu, L = None, d = None, p = None):
	r"""Compute the subspace angles between V(mu) and V(hmu)

	Defining the subspaces:

	.. math::

		\mathcal{V}(\boldsymbol{\mu}) &:=  \lbrace v_\mu \rbrace_{\mu \in \boldsymbol{\mu}} \\

	this function returns the canonical subspace angles between :math:`\mathcal{V}(\boldsymbol{\mu})` 
	and :math:`\mathcal{V}(\widehat{\boldsymbol{\mu}})`.

	Parameters
	----------
	mu: array-like (n,)
		Parameters of the subspace :math:`\mathcal{V}` where :math:`\mu_j` is in the right half plane
	hmu: array-like (m,)
		Parameters of the subspace :math:`\mathcal{M}` where :math:`\lambda_j` is in the left half plane
 

	Returns
	-------
	phi: np.array (min(n,m))
		The canonical subspace angles in radians
	"""
	mu = np.atleast_1d(np.array(mu, dtype = np.complex))
	hmu = np.atleast_1d(np.array(hmu, dtype = np.complex))
	assert np.all(mu.real > 0), "mu must be in right half plane"
	assert np.all(mu.real > 0), "hmu must be in left half plane"

	# Compute Cholesky factorization of the left matrix
	if (L is None) or (d is None) or (p is None):
		L, d, p = cauchy_ldl(mu)	

	n = len(mu)
	m = len(hmu)

	# Right matrix specifying normalization for basis
	Mhat = (np.tile(hmu.reshape(m,1), (1,m)) + np.tile(hmu.conj().reshape(1,m), (m,1)))**(-1) 
	
	# Compute central matrix
	A = (np.tile(mu.reshape(n, 1), (1,m)) + np.tile(hmu.conj().reshape(1,m), (n,1)))**(-1)

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

def cholesky_inv_norm(f, L, d, p):
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
	return np.linalg.norm(cholesky_inv(f, L, d, p),2)

class ProjectedH2MOR(H2MOR,PoleResidueSystem):
	""" Projected H2-optimal Model Reduction


	Parameters
	----------
	rom_dim: int
		Dimension of reduced order model to construct
	real: bool (optional)
		If True, fit a real dynamical system; if False, fi a complex dynamical system

	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000, verbose = False, ftol = 1e-5, cond_max= 1e20):
		H2MOR.__init__(self, rom_dim, real = real)
		self.maxiter = maxiter
		self.verbose = verbose
		self.ftol = ftol
		self.cond_max = cond_max
		self.over_determine = 2

	def _mu_init(self, H):
		if isinstance(H, StateSpaceSystem):
			lam = H.poles(which = 'LR', k = self.rom_dim+2)
			mu_imag = [np.min(lam.imag), np.max(lam.imag)]
			if self.real:
				mu_imag = np.array([-1,1])*np.max(np.abs(mu_imag))
			mu_real = -np.max(lam.real)

			# TODO: Why does using 2r+2 work better than using a recursive initialization
			# that only uses 6? 
			mu0 = mu_real + 1j*np.linspace(mu_imag[0], mu_imag[1], 2*self.rom_dim + 2)
			if self.real:
				I = marriage_sort(mu0, mu0.conjugate())
				mu0 = 0.5*(mu0 + mu0[I].conjugate())
			return mu0	
		raise NotImplementedError

	def _fit(self, H, mu0 = None):
		
		if mu0 is None:
			mu0 = self._mu_init(H)

		mu = np.array(mu0, dtype = np.complex)
		lam = np.zeros(0)	# Poles of the previous iterate
		Hr = ZeroSystem(H.output_dim, H.input_dim)
		
		# Outer loop
		for it in range(self.maxiter):
			lam_old = lam
			Hr_old = Hr
			n = len(mu)

			# Pick the order of rational approximation
			if self.real: 
				rom_dim = 2*((n-self.over_determine)//4)
				rom_dim = max(2, rom_dim)
			else: 
				rom_dim = ((n-self.over_determine)//2)
				rom_dim = max(1, rom_dim)
			rom_dim = min(self.rom_dim, rom_dim)

			# Initialize two copies of the fitting routine for the two initializations we will use 
			if self.real:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True, verbose = 0, xtol = 1e-12, gtol = 1e-10, ftol = 1e-10, max_nfev = 500) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True, verbose = 0, xtol = 1e-12, gtol = 1e-10, ftol = 1e-10, max_nfev = 500) 
			else:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True, verbose = 2) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True, verbose = 2) 

			#for Hr in [Hr1, Hr2]:
			#	Hr._transform = lambda x:x
			#	Hr._inverse_transform = lambda x:x

			# Evaluate the transfer function, recycling data
			H_mu = self.eval_transfer(H, mu)
			H_mu = H_mu.reshape(n,)		
	
			# Compute the weight matrix
			L,d,p = cauchy_ldl(mu) 
			M = lambda x: cholesky_inv(x, L, d, p)
			#Linv = M(np.eye(len(mu)))
			H_norm_est = np.linalg.norm(M(H_mu))
			# Find rational approximation (inner loop)
			# Default (AAA) initialization
			Hr1.fit(mu, H_mu, W = M)
			res_norm1 = Hr1.residual_norm()

			if (lam_old is not None) and len(lam_old) == Hr2.n:
				# Initialization based on previous poles
				Hr2.fit(mu, H_mu, W = M, lam0 = lam_old)
				res_norm2 = Hr2.residual_norm()
				# Set the reduced order model to be the smaller of the two
				if res_norm2 < res_norm1:
					Hr = Hr2
				else:
					Hr = Hr1
			else:
				Hr = Hr1
				res_norm2 = np.inf

			active_mask = np.abs(Hr._res.active_mask)
		
			lam, rho = Hr.pole_residue()	
			Hr = PoleResidueSystem(lam, rho)	

			# Poles for which constraints were not active
#			if self.real: 
#				# Due to pairwise
#				mask = np.zeros(active_mask.shape)
#				mask[0:2*(rom_dim//2):2] =  active_mask[:2*(rom_dim//2):2] + active_mask[1:2*(rom_dim//2):2]
#				mask[1:2*(rom_dim//2):2] =  active_mask[:2*(rom_dim//2):2] + active_mask[1:2*(rom_dim//2):2]
#				if rom_dim % 2 == 1:
#					mask[-1] = active_mask[-1]
#				print mask
#				lam_can = lam[ mask == 0]
#			else:
#				# only those lam for which the constraints aren't active
#				lam_can = lam[ (active_mask[::2] + active_mask[1::2]) == 0]

			# Don't allow interpolation points to be too far outside of the current interpolation points mu
			# We change lam both for generating new interpolation points
			# as well as ensuring subsequent iterations 
			lam_real = np.maximum(-2*np.max(mu.real), np.minimum(-0.5*np.min(mu.real),lam.real))
			lam_imag = np.maximum(2*np.min(mu.imag), np.minimum(2*np.max(mu.imag), lam.imag))
			lam_can = lam_real+1j*lam_imag +1e-7*1j*np.abs(lam_imag)*np.random.randn(*lam_imag.shape) 
			# If all poles are on the boundary, randomly sample from the interior of mu	
			#if len(lam_can) == 0:
				# Pick a random point in the convex hull of existing samples
				#alpha = np.random.uniform(0,1,len(mu))
				#alpha /= np.sum(alpha)
				#lam_can = -np.array([ np.dot(alpha, mu)])
			#	real_part = np.min(mu.real)
			#	lam_can = -real_part +1j*lam.imag


			# Find the largest subspace angle 
			max_angles = np.zeros(len(lam_can))
			for i in range(len(lam_can)):
				max_angles[i] = np.max(subspace_angle_V_M(mu, lam[i], L = L, d = d, p = p))
				#max_angles[i] = np.max(subspace_angle_V_V(mu, -lam_can[i].conj(), L = L, d = d, p = p))
				

			# Append existing mu
			i = np.nanargmax(max_angles)
			max_angle = max_angles[i]
			mu_star = -lam_can[i].conj()

			# Ensure in strict RHP
			mu_star = max(mu_star.real, 1e-10) + 1j*mu_star.imag
			if self.real and (mu_star.imag !=0):
				mu_star = mu_star.real + 1j*np.abs(mu_star.imag)	
			
			# Evalute termination conditions
			delta_Hr = (Hr - Hr_old).norm()/Hr.norm()
	
			# Compute the condition number
			#sigma = svdvals(np.diag(d**(0.5)).dot(L))
			U,s,VH = cauchy_hermitian_svd(mu, L = L, d = d, p = p)
			cond_M = np.max(s)/np.min(s)
	
			# Print Logging messages
			if self.verbose:
				# Header
				if it == 0:
					print("  it | dim | FOM Evals | delta Hr |  cond M  |       mu star      | res norm | max angle |  init  |")
					print("-----|-----|-----------|----------|----------|--------------------|----------|-----------|--------|")
		
				if np.abs(res_norm1 - res_norm2) < 1e-6:
					init = 'either'
				elif res_norm1 < res_norm2:
					init = 'AAA'
				else:
					init = 'lam'
				res_norm = min(res_norm1, res_norm2)	
				print("%4d | %3d |   %7d | %8.2e | %8.2e | %8.2e%+8.2ei | %8.2e | %9.6f | %6s | %12.6e" % 
					(it,rom_dim, self._total_fom_evals, delta_Hr, cond_M, mu_star.real, mu_star.imag,
					res_norm/H_norm_est, 180*max_angle/np.pi, init, (H-Hr).norm()/H.norm() ) )

			# Break if termination conditions are met
			if rom_dim == self.rom_dim:
				if cond_M > self.cond_max:
					if self.verbose:
						print("Stopped due to large condition number of M")
					break
				if delta_Hr < self.ftol:
					if self.verbose:
						print("Stopped due to small movement of Hr")
					break
		
			mu = np.hstack([mu, mu_star])
			if (np.abs(mu_star.imag) > 0) and self.real:
				mu = np.hstack([mu, mu_star.conjugate()])
		
		# Copy over to self
		PoleResidueSystem.__init__(self, lam, rho)

if __name__ == '__main__':
	from demos import build_cdplayer
	H = build_cdplayer()
	# Extract the 1/2 block
	H = H[0,1]
	Hr = ProjectedH2MOR(30, maxiter = 100, verbose = True, cond_max = 1e15, ftol = 1e-7)
	Hr.fit(H)	
	
	print("Relative H2 Norm: %12.10f" % ( (H-Hr).norm()/H.norm()))	
	

