import numpy as np
import scipy.linalg
import polyrat
from .realss import fit_real_mimo_statespace_system, pole_residue_to_real_statespace
 
	

def inner_loop(z, Y, r, poles = None):

	# run vector fitting
	if poles is None: poles = 'linearized'
	vf = polyrat.VectorFittingRationalApproximation(r-1, r, poles0 = poles)
	vf.fit(z.reshape(-1,1), Y)
	lam, R = vf.pole_residue()

	# Make rank-one approximations of each residue
	for i in range(R.shape[0]):
		U, s, VH = scipy.linalg.svd(R[i])
		R[i] = s[0]*(U[:,0:1] @ VH[0:1,:])

	# Convert into Real MIMO form
	output = pole_residue_to_real_statespace(lam, R, rank = 1)

	output = fit_real_mimo_statespace_system(z, Y, *output, stable = True)	
	# convert into format for realss
	pass
