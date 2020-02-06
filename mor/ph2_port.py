# Model reduction by port Hamiltonian systems

import numpy as np

from .system import StateSpaceSystem
from .aaa import AAARationalFit
from .marriage import hungarian_sort
from scipy.linalg import solve
from scipy.optimize import least_squares 

class PortHamiltonianSystem(StateSpaceSystem):
	def __init__(self, J, R, C, Q = None):
		# TODO Check J skew symmetric
		self._J = J
		# TODO: Check R symm. pos. semidefinite
		self._R = R
		self._C = C

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
	def __init__(self, r):
		self.r = r


	def _init(self, z, H):
		r = self.r
		H = H.flatten()
		# Use AAA to initialize if system is not provided
		aaa = AAARationalFit(self.r)
		aaa.fit(z, H)
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
			err = np.array([Hj - C.T @ solve(zj*I - J + R, C) for zj, Hj in zip(z, H)])
			err = np.hstack([err.real, err.imag])
			return err

		def jacobian(C):
			jac = np.array([ solve(zj*I - J + R, C).T + solve(zj*I + J + R, C).T for zj in z])
			return -np.vstack([jac.real, jac.imag])
	
		res = least_squares(residual, np.ones(r), jacobian) 
		C0 = res.x.reshape(1,r)
		return J0, R0, C0


	def fit(self, z, H, W = None, J0 = None, R0 = None, C0 = None):
		r"""

		Parameters
		----------
		z: array-like
			Coordinates of samples
		H: array-like
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
			J0, R0, C0 = self._init(z, H)


		theta0 = self._RJC_to_theta(R0, J0, C0)

		res = lambda theta: self.residual(theta, z, H, W)
	
		# Setup bounds so R is always has nonnegative entires	
		lb = -np.inf*np.ones(len(theta0))
		lb[0:self.r] = 0
		ub = np.inf*np.ones(len(theta0))

		res = least_squares(res, theta0, bounds = [lb, ub])

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
			


class ProjectedH2PortHamiltonian(ProjectedH2MOR):
	def __init__(self, rom_dim, maxiter = 1000, verbose = False, ftol = 1e-9):
		self.rom_dim = rom_dim
		self.real = True
		self.maxiter = maxiter
		self.verbose = verbose
		self.ftol = ftol

	def fit(self, H):
		pass


