import numpy as np

from .coordvf import coordinate_vecfit

def inner_loop(zs, ys, weights, r, Hr = None):
	r"""

	"""


	if Hr is None or Hr.state_dim != r:
		# Fit with VF
		Hr = coordinate_vecfit(zs, ys, r - 1, r,  Ms = weights, verbose = True)

	# Fit 

