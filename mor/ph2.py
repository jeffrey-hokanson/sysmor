from __future__ import division
import numpy as np
from h2mor import H2MOR


class ProjectedH2MOR(H2MOR):
	""" Projected H2-optimal Model Reduction


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


	def _fit(self, H):
		pass
