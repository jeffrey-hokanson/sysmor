# Model reduction by port Hamiltonian systems

import numpy as np

from scipy.linalg import solve, solve_triangular, svdvals
from scipy.optimize import least_squares 

from .system import StateSpaceSystem, ZeroSystem
from .aaa import AAARationalFit
from .marriage import hungarian_sort
from .h2mor import H2MOR
from .cauchy import cauchy_ldl
from .ph2 import subspace_angle_V_M


class PortHamiltonianSystem(StateSpaceSystem):
	def __init__(self, J, R, C, Q = None):
		# TODO Check J skew symmetric
		self._J = J
		# TODO: Check R symm. pos. semidefinite
		self._R = R
		n = self._R.shape[0]
		self._C = C.reshape(1,n)

		if Q is not None:
			raise NotImplementedError


	@property
	def A(self):
		return self.J - self.R
	
	@property
	def B(self):
		return self.C.T
	
	@property
	def J(self):
		return self._J

	@property
	def R(self):
		return self._R


class FitPortHamiltonianSystem(PortHamiltonianSystem):
	def __init__(self, r, verbose = 0):
		self.r = r
		self.verbose = verbose


	def _init(self, z, Hz):
		r = self.r
		Hz = Hz.flatten()
		# Use AAA to initialize if system is not provided
		aaa = AAARationalFit(self.r)
		aaa.fit(z, Hz.flatten())
		lam, res = aaa.pole_residue()

		# Ensure poles are in conjugate pairs
		I = hungarian_sort(lam, lam.conjugate())
		lam = 0.5*(lam + lam[I].conjugate())
		# ensure poles are in the LHP
		lam.real = -np.abs(lam.real)

		# Allocate storage
		R0 = np.zeros((r,r))
		J0 = np.zeros((r,r))
		C0 = np.zeros((r,))

		# Convert into matrix with same eigenvalues
		lam = lam.tolist()
		j = 0
		while len(lam)>0:
			if np.isreal(lam[0]):
				R0[j,j] = lam[0].real
				lam.pop(0)
				j += 1
			else:
				R0[j,j] = -lam[0].real 
				R0[j+1,j+1] = -lam[0].real
				J0[j,j+1] = lam[0].imag
				J0[j+1,j] = -lam[0].imag
				lam.pop(np.argmin(np.abs(lam[0].conjugate() - np.array(lam))))
				lam.pop(0)
				j += 2

		# Estimate C0 by solving a NLS problem
		I = np.eye(r)
		J = J0
		R = R0
		
		def residual(C):
			err = np.array([Hj - C.T @ solve(zj*I - J + R, C) for zj, Hj in zip(z, Hz)])
			err = np.hstack([err.real, err.imag])
			return err

		def jacobian(C):
			jac = np.array([ solve(zj*I - J + R, C).T + solve(zj*I + J + R, C).T for zj in z])
			return -np.vstack([jac.real, jac.imag])
	
		res = least_squares(residual, np.ones(r), jacobian)
		C0 = res.x.reshape(1,r)
		return J0, R0, C0


	def fit(self, z, Hz, W = None, J0 = None, R0 = None, C0 = None):
		r"""

		Parameters
		----------
		z: array-like
			Coordinates of samples
		Hz: array-like
			Evaluations of the transfer function
		W: function
			Evaluates the weight matrix  
		J0: array-like (r, r)
			Upper triangular part of skew part
		R0: array-like (r,r)
		

			Symmetric positive semidefinite part
		C0: array-like (r,)
			input/output component
		"""
		if W is None:
			W = lambda x: x

		if J0 is None or R0 is None or C0 is None:
			J0, R0, C0 = self._init(z, Hz)
			
		theta0 = self._RJC_to_theta(R0, J0, C0)

		res = lambda theta: self.residual(theta, z, Hz, W)
	
		# Setup bounds so R is always has nonnegative entires	
		lb = -np.inf*np.ones(len(theta0))
		lb[0:self.r] = 0
		ub = np.inf*np.ones(len(theta0))

		res = least_squares(res, theta0, bounds = [lb, ub], verbose = self.verbose)

		theta = res.x
		R, J, C = self._theta_to_RJC(theta)
		PortHamiltonianSystem.__init__(self, J, R, C)

	def _theta_to_RJC(self, theta):
		r = self.r
		R = np.diag(theta[0:r])
		J = np.zeros((r,r))
		J[np.triu_indices(r,1)] = theta[r:r+(r*(r-1))//2]
		J[np.tril_indices(r,-1)] = -theta[r:r+(r*(r-1))//2]
		C = theta[r+(r*(r-1))//2:].reshape(r)
		return R, J, C

	def _RJC_to_theta(self, R, J, C):
		r = self.r
		return np.hstack([ np.diag(R), J[np.triu_indices(r,1)], C.flatten()])

	def residual(self, theta, z, H, W):
		H = H.flatten()
		
		R, J, C = self._theta_to_RJC(theta)
		I = np.eye(self.r)
		err = np.array([Hj - C.T @ solve(zj*I - J + R, C) for zj, Hj in zip(z, H)])
		err = W(err)
		err = np.hstack([err.real, err.imag])
		return err

	def jacobian(self, theta, z, H, W):
		H = H.flatten()
		R, J, C = self._theta_to_RJC(theta)
		r = self.r
		I = np.eye(r)
	
		jac = np.zeros((len(H), len(theta)), dtype = np.complex)
		for j in range(len(H)):
			zj = z[j]
			# matrix term
			ATc = solve(zj*I + J + R, C)
			Ac = solve(zj*I - J + R, C)
			mat = np.outer(ATc, Ac)
			#mat = ATc.reshape(r,1) @ Ac.reshape(1,r)
			jac[j, 0:r] = -np.diag(mat)
			jac[j, r:r + ((r*(r-1))//2)] = mat[np.triu_indices(r,1)] - mat[np.tril_indices(r,-1)]
			jac[j, r + ((r*(r-1))//2):] = ATc + Ac
		jac = W(jac)
		return -np.vstack([jac.real, jac.imag])
			


class ProjectedH2Generic(H2MOR):
	r""" Construct a reduced order model usting the projected nonlinear least squares framework
	"""
	def __init__(self, maxiter = 1000, verbose = False, ftol = 1e-9, M_cond_max = 1e16):
		self.maxiter = int(maxiter)
		assert self.maxiter > 0, "Maximum number of iterations must be positive"
		
		self.verbose = verbose
		self.ftol = float(ftol)
		self.M_cond_max = float(M_cond_max)
		assert self.ftol > 0, "Convergence tolerance must be positive"

		self._init_logging()

class ProjectedH2StateSpace(ProjectedH2Generic):
	r""" Generic handler for a state-space ROM

	Regardless of the constraints on the ROM, if it has a state-space form
	the optimality conditions are a subspace of those of an arbitrary state-space ROM.
	Hence, we can use the same update heuristic for :math:`\mu`.
	
	""" 	
	def __init__(self, rom_dim, maxiter = 1000, verbose = False, ftol = 1e-9, real = False, growth = 10, M_cond_max = 1e16):
		self.rom_dim = int(rom_dim)
		self.real = real
		self.growth = float(growth)
		ProjectedH2Generic.__init__(self, maxiter = maxiter, verbose = verbose, ftol = ftol, M_cond_max = M_cond_max)

	def _mu_init(self, H):
		if isinstance(H, StateSpaceSystem):
			# Make sure overdetermined
			# TODO: What if there aren't enough poles?
			lam = H.poles(which = 'LR', k = 2*self.rom_dim+2 )
			mu0 = np.abs(lam.real) + 1j*lam.imag
			if self.real:
				I = hungarian_sort(mu0, mu0.conjugate())
				mu0 = 0.5*(mu0 + mu0[I].conjugate())
			return mu0	
		raise NotImplementedError

	def _fit(self, H, mu0 = None):
	
		if mu0 is None:
			mu0 = self._mu_init(H)
		mu = np.array(mu0, dtype = np.complex)
		# NB: this caches evaluations of the transfer function
		H_mu = self.eval_transfer(H, mu)

		Hr = ZeroSystem(1,1)
		for it in range(self.maxiter):
			Hr_old = Hr
		
			# Compute the weight matrix
			L,d,p = cauchy_ldl(mu)
			# Compute the condition number
			s = svdvals(L @ np.diag(np.sqrt(d)))**2 
			M_cond = s[0]/s[-1]

			# Construct ROM
			Hr, res_norms = self._fit_rom(H, mu, Hr_old, L, d, p)

			# Pick new poles
			mu, max_angle = self._mu_update(mu, H, Hr, L, d, p)			

			# Check convergence critera
			delta_Hr = (Hr - Hr_old).norm()

			# Print convergence information
			self._print_logging(H, Hr, mu, it, res_norms, delta_Hr, M_cond, max_angle, L, d, p)

			if delta_Hr < self.ftol:
				break

			if M_cond > self.M_cond_max:
				break	
		
		# Copy over solution
		self._post_fit(Hr)	

	def _fit_rom(self, H, mu, Hr, L, d, p):	
		raise NotImplementedError

	def _mu_update(self, mu, H, Hr, L, d, p):

		# Get backup poles from AAA
		H_mu = self.eval_transfer(H, mu)
		aaa = AAARationalFit(self.rom_dim)
		aaa.fit(mu, H_mu.flatten())
		lam_aaa, _ = aaa.pole_residue()
		# Flip poles
		lam_aaa.real = -np.abs(lam_aaa.real)

		# Compute poles
		lam, rho = Hr.pole_residue()
		I = np.argsort(-lam.imag)
		lam = lam[I]
		rho = rho[I]
			
		# sort the AAA poles to match
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

			
		mu = np.hstack([mu, mu_star])
		if (np.abs(mu_star.imag) > 0) and self.real:
			mu = np.hstack([mu, mu_star.conjugate()])
		return mu, max_angle

	
	def _print_logging(self, H,  Hr, mu, it, res_norms, delta_Hr, M_cond, max_angle, L, d, p):


		if it == 0 or self.verbose >= 10:
			head1 = "  it | dim | FOM Evals | delta Hr |  cond M  |       mu star      | res norm | max angle |  init  |"
			head2 = "-----|-----|-----------|----------|----------|--------------------|----------|-----------|--------|"
	#		if self.print_norm:
	#			head1 += ' Error H2 Norm |'
	#			head2 += '---------------|'
			print(head1)
			print(head2)
			
		res_norm1, res_norm2 = res_norms
	
		if np.abs(res_norm1 - res_norm2)/min([res_norm1,res_norm2]) < 1e-6:
			init = 'either'
		elif res_norm1 < res_norm2:
			init = 'AAA'
		else:
			init = 'lam'
		
		mu_star = mu[-1]
		res_norm = min(res_norm1, res_norm2)
		rom_dim = Hr.A.shape[0]
		M = lambda x: np.diag(d**(-0.5)) @ solve_triangular(L, x[p], lower = True, trans = 'N')
		H_mu = self.eval_transfer(H, mu).flatten()
		H_norm_est = np.linalg.norm( M(H_mu))
		iter_message = "%4d | %3d |   %7d | %8.2e | %8.2e | %8.2e%+8.2ei | %8.2e | %9.6f | %6s |" % \
			(it, self.rom_dim, self._total_fom_evals, delta_Hr, M_cond, mu_star.real, mu_star.imag,
			res_norm/H_norm_est, 180*max_angle/np.pi, init )
		print(iter_message)
		


class ProjectedH2PortHamiltonian(ProjectedH2StateSpace, PortHamiltonianSystem ):
	def __init__(self, rom_dim, maxiter = 1000, verbose = False, ftol = 1e-9):
		ProjectedH2StateSpace.__init__(self, rom_dim, maxiter = maxiter, verbose = verbose, ftol = ftol, real = True)


	def _fit_rom(self, H, mu, Hr_old, L, d, p):
		M = lambda x: np.diag(d**(-0.5)) @ solve_triangular(L, x[p], lower = True, trans = 'N')

		# Try AAA-based initialization
		H_mu = self.eval_transfer(H, mu).flatten()

		Hr1 = FitPortHamiltonianSystem(self.rom_dim, verbose = 0)
		Hr1.fit(mu, H_mu,  W = M)
		Hr1_mu = np.array([Hr1.transfer(mu_j) for mu_j in mu]).flatten()
		res_norm1 = np.linalg.norm( M(H_mu - Hr1_mu))


		# Try previous iterate initialization
		Hr2 = FitPortHamiltonianSystem(self.rom_dim, verbose = 0)
		try:
			J0 = Hr_old.J
			R0 = Hr_old.R
			C0 = Hr_old.C
			Hr2.fit(mu, H_mu, W = M, J0 = J0, R0 = R0, C0 = C0) 		
			Hr2_mu = np.array([Hr2.transfer(mu_j) for mu_j in mu]).flatten()
			res_norm2 = np.linalg.norm( M(H_mu - Hr2_mu))
		except AttributeError as e:
			res_norm2 = np.inf

		if res_norm1 < res_norm2:
			Hr = Hr1
		else:
			Hr = Hr2

		return Hr, (res_norm1, res_norm2)

	def _post_fit(self, Hr):
		PortHamiltonianSystem.__init__(self, Hr.J, Hr.R, Hr.C)

	
