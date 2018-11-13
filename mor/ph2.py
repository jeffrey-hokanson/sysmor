from __future__ import division
import numpy as np
from h2mor import H2MOR
from pffit import PartialFractionRationalFit
from cauchy import cauchy_ldl, cauchy_inv_norm  

def subspace_angle_V_M(mu, lam):
	"""Compute the subspace angles between V and M

	
	"""
	pass


class ProjectedH2MOR(H2MOR):
	""" Projected H2-optimal Model Reduction


	Parameters
	----------
	rom_dim: int
		Dimension of reduced order model to construct
	real: bool (optional)
		If True, fit a real dynamical system; if False, fi a complex dynamical system

	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000):
		H2MOR.__init__(self, rom_dom, real = real)
		self.maxiter = maxiter

	def _mu_init(self, H):
		raise NotImplementedError

	def _fit(self, H, mu0 = None):

		if mu0 is None:
			self._mu_init(H)

		mu = np.copy(mu0)
		lam_old = None		# Poles of the previous iterate

		# Outer loop
		for it in range(self.maxiter):
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
			
			# Compute the weight matrix
			L,d,p = cauchy_ldl(mu) 
			M = lambda x: cauchy_inv_norm(x, L, d, p)

			# Find rational approximation (inner loop)

			# Default (AAA) initialization
			Hr1.fit(mu, H_mu, W = M)

			# Initialization based on previous poles
			if (lam_old is not None) and len(lam_old) == Hr2.rom_dim:
				Hr2.fit(mu, H_mu, W = M, lam0 = lam_old)
				if Hr2.residual_norm() < Hr1.residual_norm()
					Hr = Hr2
				else:
					Hr = Hr1
			else:
				Hr = Hr1

			lam_old, _ = Hr.pole_residue()	

			# Check termination conditions


			# Update mu


			break
