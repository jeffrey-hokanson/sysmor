from __future__ import division
import numpy as np



class H2MOR:


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
			If :code:`der=True`, the derivative of :code:`H` at :code:`mu`

		"""
		H_mu = np.nan*np.zeros(mu.shape, dtype = np.complex)
		for i, mu_i in enumerate(mu):
			
			if not np.isfinite(H_mu[i]):
				H_mu[i] = complex(H.transfer(mu_))
				# Find the conjugate point if included in this set of samples
				if self.real:
					I = np.argwhere( np.abs(mu - mu_.conj())/np.abs(mu_) < 1e-12)
					H_mu[I] = H_mu[i].conjugate()
				self.total_fom_evals += 1
		return H_mu
	
	
	def fit(self, H, **kwargs):
		""" Construct a reduced order model

		Parameters
		----------
		H: LTISystem
			Full order model to reduce 
		"""
		# Initialize dictionary of existing samples of H
		self._H_mu = {}
		self._H_mu_der = {}
		# Initialize history 
		self.history = []

		# Call child fit routine
		self._fit(H, **kwargs)
