from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular, cholesky, svdvals
from warnings import catch_warnings
from .system import StateSpaceSystem, PoleResidueSystem, ZeroSystem
from .h2mor import H2MOR
from .pffit import PartialFractionRationalFit
from .cauchy import cauchy_ldl, cauchy_hermitian_svd 
from .marriage import hungarian_sort
from .subspace import subspace_angle_V_M_mp
from .aaa import AAARationalFit
from .vecfit import VFRationalFit

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
	assert np.all(mu.real >= 0), "mu must be in right half plane"
	assert np.all(lam.real <= 0), "lam must be in left half plane"

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
	maxiter: int
		Maximum number of iterations, each costing one evaluation of H, to take
	verbose: bool or int
		Specify verbosity level:
		* False: do not print
		* True : print convergence history
		* >= 10: print choices of mu_star and corresponding subspace angles
		* >=100: print convergence history of inner loop
	ftol: float, positive
		Tolerance for change in successive iterations of the reduced order model
	cond_max: float, positive
		Maximum condition number of M before iteration terminates
	growth: float, positive
		
	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000, verbose = False, ftol = 1e-9, 
		cond_max= 1e18, growth = 10, print_norm = False, spectral_abscissa = -1e-6):
		H2MOR.__init__(self, rom_dim, real = real)
		self.maxiter = maxiter
		self.verbose = verbose
		self.ftol = ftol
		self.cond_max = cond_max
		self.over_determine = 2
		self.print_norm = print_norm
		self.growth = growth
		self._spectral_abscissa = spectral_abscissa

	def _mu_init(self, H):
		if isinstance(H, StateSpaceSystem):
			#lam = H.poles(which = 'LR', k= 6)
			#lam = H.poles(which = 'LR', k= max(6,self.rom_dim) )
			lam = H.poles(which = 'LR', k = max(self.rom_dim, 4) )
			#mu0 = np.abs(lam.real)+ 1j*lam.imag
			#mu_imag = [np.min(lam.imag), np.max(lam.imag)]
			#if self.real:
			#	mu_imag = np.array([-self.growth,self.growth])*np.max(np.abs(mu_imag))
			mu0 = np.abs(lam.real) + 1j*lam.imag
			#mu_real = -np.max(lam.real)
			#mu0 = mu_real + 1j*np.linspace(mu_imag[0], mu_imag[1], 6)
			#mu0 = mu_real + 1j*np.linspace(mu_imag[0], mu_imag[1], 2*self.rom_dim+2)
			#mu0 = mu_real + 1j*lam.imag
			if self.real:
				I = hungarian_sort(mu0, mu0.conjugate())
				mu0 = 0.5*(mu0 + mu0[I].conjugate())
			return mu0	
		raise NotImplementedError

	def _fit(self, H, mu0 = None):
		# If a spectral abscissa hasn't been provided, use that of the FOM
		#if self._spectral_abscissa is None:
		#	try:
		#		self._spectral_abscissa = H.spectral_abscissa()
		#	except:
		#		pass	
		self._spectral_abscissa = 0
	
		if mu0 is None:
			mu0 = self._mu_init(H)
		mu = np.array(mu0, dtype = np.complex)
		Hr = ZeroSystem(H.output_dim, H.input_dim)

		lam_proj = None
		valid_poles = np.array([])		
		lam = []

		# Outer loop
		for it in range(self.maxiter):
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

			# Compute the weight matrix
			L,d,p = cauchy_ldl(mu)

			# Compute the condition number
			s = svdvals(L @ np.diag(np.sqrt(d)))**2 

			if np.min(s) == 0:
				if self.verbose:
					print("Weight matrix singular")
					print(s)	
					print('mu')
					for mu_i in mu:
						print(mu_i)
				break

			cond_M = np.max(s)/np.min(s)
			if rom_dim == self.rom_dim and (cond_M > self.cond_max):
				if self.verbose:
					print("Stopped due to large condition number of M")
				break
			
			M = lambda x: cholesky_inv(x, L, d, p)
			
			# Evaluate the transfer function, recycling data
			H_mu = self.eval_transfer(H, mu)
			H_mu = H_mu.reshape(n,)		
	
			try:	
				H_norm_est = np.linalg.norm(M(H_mu))
			except (ValueError,np.linalg.linalg.LinAlgError):
				H_norm_est = np.inf
	
			###################################################################
			# Find rational approximation (inner loop)
			###################################################################
			
			# Initialize two copies of the fitting routine for the two initializations we will use 
			kwargs = {}
			kwargs['xtol'] = 3e-16
			kwargs['gtol'] = 3e-16
			kwargs['ftol'] = 3e-16
			kwargs['max_nfev'] = 10*self.rom_dim
			#if self._spectral_abscissa is not None:
			#	kwargs['spectral_abscissa'] = self._spectral_abscissa
			if self.verbose >= 100:
				kwargs['verbose'] = 2

			if self.real:	
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True, **kwargs) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'real', stable = True, **kwargs) 
			else:			
				Hr1 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True, **kwargs) 
				Hr2 = PartialFractionRationalFit(rom_dim-1, rom_dim, field = 'complex', stable = True, **kwargs) 
			
			# Default (AAA) initialization
			aaa = AAARationalFit(rom_dim)
			aaa.fit(mu, H_mu)
			lam_aaa, _ = aaa.pole_residue()
			# Flip poles
			lam_aaa.real = -np.abs(lam_aaa.real)

			try:
				Hr1.fit(mu, H_mu, W = M, lam0 = lam_aaa)
				res_norm1 = Hr1.residual_norm()
			except (ValueError, np.linalg.linalg.LinAlgError):
				res_norm1 = np.inf

			# Initialization based on previous poles
			if len(lam) == rom_dim:
				#vf = VFRationalFit(rom_dim - 1, rom_dim, W = M)
				#vf.fit(mu, H_mu)
				#lam_vf, rho_vf = vf.pole_residue()
				Hr2.fit(mu, H_mu, W = M, lam0 = lam)
				res_norm2 = Hr2.residual_norm()
			else:
				res_norm2 = np.inf

			if not (np.isfinite(res_norm1) or np.isfinite(res_norm2)):
				if verbose: 
					print("Both initializations failed")
				break
		
			if res_norm1 < res_norm2:
				res_norm = res_norm1
				Hr = Hr1
			else:
				res_norm = res_norm2
				Hr = Hr2

			###################################################################
			# Update the list of valid poles
			# This is used for initialization of the next iteration
			###################################################################

			# Convert into a pole-residue representation
			lam, rho = Hr.pole_residue()
			I = np.argsort(-lam.imag)
			lam = lam[I]
			rho = rho[I]
			# sort the aaa poles to match
			I = hungarian_sort(lam, lam_aaa)
			lam_aaa = lam_aaa[I]
		
			# Determine which poles are valid	
			valid = np.ones(len(lam), dtype = np.bool)

			real_left = np.min(-mu.real)
			valid = valid & (lam.real >= real_left * self.growth)
			
			real_right = np.max(-mu.real)
			valid = valid & (lam.real <= real_right / self.growth)

			imag_top = np.max(mu.imag)
			valid = valid & (lam.imag <= imag_top * self.growth)

			imag_bot = np.min(mu.imag)
			valid = valid & (lam.imag >= imag_bot * self.growth)


			# For poles to be valid, they must also be well separated
			for i in range(len(lam)):
				valid[i] = valid[i] & np.all(abs(lam[i] - lam[0:i])>1e-8) & np.all(abs(lam[i] - lam[i+1:]) > 1e-8)

			

			###################################################################
			# Choose new interpolation point
			###################################################################

			lam_can = np.copy(lam)
			# Use AAA poles when a pole of Hr is invalid			
			lam_can[~valid] = lam_aaa[~valid]

			# Compute the subspace angle for each
			max_angles = np.nan*np.zeros(len(lam))
			for i in range(len(lam)):
				max_angles[i] = np.max(subspace_angle_V_M(mu, lam_can[i], L = L, d = d, p = p))

			while True:
				try:	
					k = np.nanargmax(max_angles)
				except ValueError as e:
					print("No valid new shifts found")
					raise e

				mu_star = -lam_can[k].conj()
				max_angle = max_angles[k]
				# Ensure we have a distinct mu
				if np.min(np.abs(mu_star - mu)) == 0:
					max_angles[k] = np.nan
				else:
					break			

	

			if self.verbose >= 10:
				print("")
				for i in range(len(lam)):
					line = 'angle %10.4f | ' % (180/np.pi*max_angles[i])
					line += 'lam %+5.2e  %+5.2e I | ' % (lam[i].real, lam[i].imag)
					line += 'rho %+5.2e  %+5.2e I | ' % (rho[i].real, rho[i].imag)
					line += 'lam_can %+5.2e  %+5.2e I | ' % (lam_can[i].real, lam_can[i].imag)
					line += 'lam_aaa %+5.2e  %+5.2e I | ' % (lam_aaa[i].real, lam_aaa[i].imag)

					if valid[i]:
						line += '   '
					else:
						line += ' X '

					if i == k:
						line += " <=== "
					else:
						line += "      "
				
					print(line)
				print("")

			

			###################################################################
			# Evalute termination conditions
			###################################################################
			
			lam[~valid] = lam_aaa[~valid]
			rho[~valid] = 0.
			Hr = PoleResidueSystem(lam, rho)	
			Hr_norm = Hr.norm()
			delta_Hr = (Hr - Hr_old).norm()/Hr_norm
			
			###################################################################
			# Print Logging messages
			###################################################################
			if self.verbose:
				# Header
				if it == 0 or self.verbose >= 10:
					head1 = "  it | dim | FOM Evals | delta Hr |  cond M  |       mu star      | res norm | max angle |  init  |"
					head2 = "-----|-----|-----------|----------|----------|--------------------|----------|-----------|--------|"
					if self.print_norm:
						head1 += ' Error H2 Norm |'
						head2 += '---------------|'
					print(head1)
					print(head2)
	
				if np.abs(res_norm1 - res_norm2)/min([res_norm1,res_norm2]) < 1e-6:
					init = 'either'
				elif res_norm1 < res_norm2:
					init = 'AAA'
				else:
					init = 'lam'
				res_norm = min(res_norm1, res_norm2)	
				iter_message = "%4d | %3d |   %7d | %8.2e | %8.2e | %8.2e%+8.2ei | %8.2e | %9.6f | %6s |" % \
					(it,rom_dim, self._total_fom_evals, delta_Hr, cond_M, mu_star.real, mu_star.imag,
					res_norm/H_norm_est, 180*max_angle/np.pi, init )

				if self.print_norm:
					iter_message += ' %13.6e |' % ( (H-Hr).norm()/H.norm())

				print(iter_message)
			
			###################################################################
			# Copy logging information
			###################################################################
			if self.history is not None:
				# TODO: Do we need to copy Hr? I don't think so since there is no way to edit it in place
				self.history.append({
					'mu': np.copy(mu), 
					'Hr': Hr, 
					'total_fom_evals': self._total_fom_evals,
					'total_fom_der_evals': self._total_fom_der_evals,
					'total_linear_solves': self._total_linear_solves,
				})


			###################################################################
			# Break if termination conditions are met
			###################################################################
			if rom_dim == self.rom_dim:
				if delta_Hr < self.ftol/Hr_norm:
					if self.verbose:
						print("Stopped due to small movement of Hr")
					break
		
			###################################################################
			# Update projector
			###################################################################
			mu = np.hstack([mu, mu_star])
			if (np.abs(mu_star.imag) > 0) and self.real:
				mu = np.hstack([mu, mu_star.conjugate()])
		
		# Copy over to self
		PoleResidueSystem.__init__(self, lam, rho)

if __name__ == '__main__':
	from demos import build_iss

	H = build_iss()
	# Extract the 1/2 block
	H = H[0,0]
	Hr = ProjectedH2MOR(40, maxiter = 100, verbose = True, cond_max = 1e15, ftol = 1e-9, print_norm = True)
	Hr.fit(H)	
	
	print("Relative H2 Norm: %12.10f" % ( (H-Hr).norm()/H.norm()))	
	

