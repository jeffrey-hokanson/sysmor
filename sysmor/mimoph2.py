import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator
import polyrat
import iterprinter

from .realss import fit_real_mimo_statespace_system, pole_residue_to_real_statespace
from .cauchy import cauchy_ldl 


class Weight(LinearOperator):
	r""" Compute the action of the weight matrix associated with a set of projectors
	"""
	def __init__(self, mu):
		self.mu = mu
		self.L, self.d, self.p = cauchy_ldl(mu)		

	@property 
	def shape(self):
		return self.L.shape

	def _matmat(self, X):
		LinvX = scipy.linalg.solve_triangular(self.L, X[self.p], lower = True, trans = 'N')
		# np.diag(self.d**(-0.5)) @ LinvX
		return np.multiply( 1./np.sqrt(self.d[:,None]), LinvX)

	

def inner_loop(z, Y, r, weight, poles = None):
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
	# It seems more expedient to explicitly form the inverse
	# this is not unreasonable given how many times we apply it.
	if True:
		weight = weight @ np.eye(Y.shape[0])

	vf = polyrat.VectorFittingRationalApproximation(r-1, r, poles0 = poles, verbose = False, maxiter = 100)
	vf.fit(z.reshape(-1,1), Y)
	lam, R = vf.pole_residue()

	# Convert into Real MIMO form with rank-one residues
	output = pole_residue_to_real_statespace(lam, R, rank = 1)

	# Optimize the fit 
	H = fit_real_mimo_statespace_system(z, Y, *output, stable = True, weight = weight)
	return H

def score_candidates(weight, mu, mu_can):
	score = np.zeros(mu_can.shape)
	m = len(mu)
	for k, muc in enumerate(mu_can):
		x = (mu + muc.conj())**(-1)
		Mc = (muc + muc.conj())**(-1)
		x /= np.sqrt(Mc)
		wx = weight @ x
		score[k] = np.linalg.norm(wx)
	return score

def project_mu(mu, mu_can, scale = 10):
	mu_can.real = np.minimum(mu_can.real, scale*np.max(mu.real))
	mu_can.real = np.maximum(mu_can.real, (1/scale)*np.min(mu.real))
	mu_can.imag = np.minimum(mu_can.imag, scale*np.max(mu.imag))
	mu_can.imag = np.maximum(mu_can.imag, scale*np.min(mu.imag))
	return mu_can


def outer_loop(H, r, mu0, maxiter = 100):
	mu = np.copy(mu0.astype(np.complex))
	Hmu = H.transfer(mu)
	poles = None
	it = 0
	while True:
		weight = Weight(mu) @ np.eye(Hmu.shape[0])
		Hr = inner_loop(mu, Hmu, r, weight)
		poles = Hr.poles()
		mu_can = np.abs(poles.real) + 1j*poles.imag
		mu_can = project_mu(mu, mu_can)

		err_norm = (H - Hr).norm()
		print("norm", err_norm)

		score = score_candidates(weight, mu, mu_can)
		k = np.argmax(score)

		it += 1

		if it > maxiter: break 
		
		Hnew = H.transfer(mu_can[k])
		
		if np.isreal(mu_can[k]):
			mu = np.hstack([mu, mu_can[k]])
			Hmu = np.concatenate([Hmu, Hnew])
		else:
			mu = np.hstack([mu, mu_can[k], mu_can[k].conj()] )
			Hmu = np.concatenate([Hmu, Hnew, Hnew.conj()] )
				


