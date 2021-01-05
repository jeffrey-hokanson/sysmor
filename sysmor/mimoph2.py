import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator
import polyrat
from .realss import fit_real_mimo_statespace_system, pole_residue_to_real_statespace
from .cauchy import cauchy_ldl 


class Weight(LinearOperator):
	r""" Compute the action of the weight matrix associated with a set of projectors
	"""
	def __init__(self, mu):
		self.L, self.d, self.p = cauchy_ldl(mu)		

	@property 
	def shape(self):
		return self.L.shape

	def _matmat(self, X):
		LinvX = scipy.linalg.solve_triangular(self.L, X[self.p], lower = True, trans = 'N')
		# np.diag(self.d**(-0.5)) @ LinvX
		return np.multiply( 1./np.sqrt(self.d[:,None]), LinvX)

	

def inner_loop(z, Y, r, poles = None):
	r"""
	Note: this assumes full data, not tangential.
	"""
	# run vector fitting
	if poles is None: 
		poles = 'linearized'
	elif len(poles) < r:
		raise NotImplementedError

	# The z's are the points at which we evaluate the transfer function
	# in the RHP (also mu's in the text) 
	weight = Weight(z) 
	# It seems more expedient to explicitly form the inverse
	# this is not unreasonable given how many times we apply it.
	if True:
		weight = weight @ np.eye(Y.shape[0])

	vf = polyrat.VectorFittingRationalApproximation(r-1, r, poles0 = poles)
	vf.fit(z.reshape(-1,1), Y)
	lam, R = vf.pole_residue()

	# Make rank-one approximations of each residue
	#for i in range(R.shape[0]):
	#	U, s, VH = scipy.linalg.svd(R[i])
	#	R[i] = s[0]*(U[:,0:1] @ VH[0:1,:])

	# Convert into Real MIMO form with rank-one residues
	output = pole_residue_to_real_statespace(lam, R, rank = 1)

	# Optimize the fit 
	output = fit_real_mimo_statespace_system(z, Y, *output, stable = True, weight = weight)
	
