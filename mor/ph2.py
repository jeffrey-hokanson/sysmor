from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular, cholesky, svdvals
from system import StateSpaceSystem, PoleResidueSystem, ZeroSystem
from h2mor import H2MOR
from pffit import PartialFractionRationalFit
from cauchy import cauchy_ldl 
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
	Mhat = (np.tile(hmu.conj().reshape(m,1), (1,m)) + np.tile(hmu.reshape(1,m), (m,1)))**(-1) 
	

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
		self.over_determine = 1

	def _mu_init(self, H):
		if isinstance(H, StateSpaceSystem):
			lam = H.poles(which = 'LR', k = 2)
			mu_imag = [np.min(lam.imag), np.max(lam.imag)]
			if self.real:
				mu_imag = np.array([-1,1])*np.max(np.abs(mu_imag))
			mu_real = -np.max(lam.real)

			mu0 = mu_real + 1j*np.linspace(mu_imag[0], mu_imag[1], 6)
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
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True) 
			else:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True) 

			for Hr in [Hr1, Hr2]:
				Hr._transform = lambda x:x
				Hr._inverse_transform = lambda x:x

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
			residual_norm = Hr.residual_norm()

			lam, rho = Hr.pole_residue()	
			Hr = PoleResidueSystem(lam, rho)	

			# Find the largest subspace angle 
			max_angles = np.zeros(len(lam))
			max_angles2 = np.zeros(len(lam))
			for i in range(len(lam)):
				max_angles[i] = np.max(subspace_angle_V_M(mu, lam[i], L = L, d = d, p = p))
			
			# Append existing mu
			i = np.argmax(max_angles)
			mu_star = -lam[i].conj()
			# Ensure in strict RHP
			mu_star = max(mu_star.real, 1e-10) + 1j*mu_star.imag
			if self.real:
				mu_star = mu_star.real + 1j*np.abs(mu_star.imag)	
			
			# Evalute termination conditions
			delta_Hr = (Hr - Hr_old).norm()/Hr.norm()
	
			# Compute the condition number
			sigma = svdvals(np.diag(d**(0.5)).dot(L))
			cond_M = (np.max(sigma)/np.min(sigma))**2
	
			# Print Logging messages
			if self.verbose:
				# Header
				if it == 0:
					print("  it | dim | FOM Evals | delta Hr |  cond M  |       mu star      | res norm |")
					print("-----|-----|-----------|----------|----------|--------------------|----------|")
				print("%4d | %3d |   %7d | %8.2e | %8.2e | %8.2e%+8.2ei | %8.2e | %8.2e" % 
					(it,rom_dim, self._total_fom_evals, delta_Hr, cond_M, mu_star.real, mu_star.imag,
					residual_norm, (H-Hr).norm()/H.norm() ) )

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
	Hr = ProjectedH2MOR(22, maxiter = 100, verbose = True, cond_max = 1e14, ftol = 1e-7)
	Hr.fit(H)	
	
	print("Relative H2 Norm: %5.2e" % ( (H-Hr).norm()/H.norm()))	
	

