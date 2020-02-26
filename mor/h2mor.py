from __future__ import division
import numpy as np



class H2MOR:
	""" Abstract Base Class for H2 Model Reduction

	Parameters
	----------
	rom_dim: int
		Dimension of reduced order model to construct
	real: bool (optional)
		If True, fit a real dynamical system; if False, fi a complex dynamical system

	"""
	def __init__(self, rom_dim, real = True):
		self.rom_dim = rom_dim
		self.real = True
		self._init_logging()

	def _init_logging(self):
		self._total_fom_evals = 0
		self._total_fom_der_evals = 0
		self._total_linear_solves = 0		
		# Initialize dictionary of existing samples of H
		self._H_mu = {}
		self._H_mu_der = {}

		# Initialize history 
		self.history = []

	def eval_transfer(self, H, mu, der = False):
		""" Helper function to evaluate the transfer function, recycling existing data.

		Given a transfer function represented by :math:`H \in \mathcal{H}_2`,
		evaluate the transfer function at points :math:`\\boldsymbol{\mu}\in \mathbb{C}^n`.

		Parameters
		----------
		H: LTISystem
			System with transfer function to evaluate
		mu: array-like
			Points at which to evaluate the transfer function
		der: bool
			If true, include the derivative as well.
		
		Returns
		-------
		H_mu: array-like
			Value of the transfer function at :code:`mu`
		H_mu_der: array-like
			If :code:`der=True`, the derivative of :code:`H` at :code:`mu`; otherwise not provided.

		"""
		H_mu = np.nan*np.zeros((len(mu), H.input_dim, H.output_dim), dtype = np.complex)
		if der:
			H_mu_der = np.nan*np.zeros(mu.shape, dtype = np.complex)
	
		H_real = H.isreal
	
		for i, mu_i in enumerate(mu):
			# Try to pull function values from cache
			if True:
				try: H_mu[i] = self._H_mu[mu_i]
				except KeyError: pass

			if H_real and np.any(np.isnan(H_mu[i])): 
				try: H_mu[i] = self._H_mu[mu_i.conj()].conj()
				except KeyError: pass
			
			if der:
				try: H_mu_der[i] = self._H_mu_der[mu_i]
				except KeyError: pass

			if der and H_real and np.any(np.isnan(H_mu_der[i])):
				try: H_mu_der[i] = self._H_mu_der[mu_i.conj()].conj()
				except KeyError: pass
			
			# For those points not in the cache, evaluate
			if (not der) and np.any(np.isnan(H_mu[i])):
				H_mu[i] = H.transfer(mu_i, der = False)
				self._H_mu[mu_i] = np.copy(H_mu[i])
				self._total_fom_evals += 1

			if der and (np.any(np.isnan(H_mu[i])) or np.any(np.isnan(H_mu_der[i]))):
				H_mu[i], H_mu_der[i] = H.transfer(mu_i, der = True)
				self._H_mu[mu_i] = np.copy(H_mu[i])
				self._H_mu_der[mu_i] = np.copy(H_mu_der[i])
				self._total_fom_evals += 1
				self._total_fom_der_evals += 1


		if der:
			return H_mu, H_mu_der
		else:
			return H_mu
	
	
	def fit(self, H, **kwargs):
		""" Construct a reduced order model

		Given a linear time invariant system :math:`H`, 
		construct a reduced order model of the specified degree.


		Parameters
		----------
		H: LTISystem
			Full order model to reduce 
		"""

		# Re-initialize data structures for logging
		self._init_logging()

		# Call child fit routine
		self._fit(H, **kwargs)


