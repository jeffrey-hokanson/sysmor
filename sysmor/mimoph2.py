import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator
import polyrat
import iterprinter

from .realss import fit_real_mimo_statespace_system, pole_residue_to_real_statespace
from .cauchy import cauchy_ldl 

from colorama import Fore, Style

def _norm(weight, x):
	r""" Evaluate the Frobenius norm where weight is applied to each x[:,*idx]
	"""
	M = x.shape[0]
	if weight is None:
		weight = _identity(M)

	norm = 0.
	for idx in np.ndindex(x.shape[1:]):
		norm += np.sum( np.abs(weight @ x[(slice(M), *idx)])**2)

	return np.sqrt(norm)


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

	@property
	def cond(self):
		s = scipy.linalg.svdvals(self.L @ np.diag(np.sqrt(self.d)))**2
		return np.max(s)/np.min(s)

	@property
	def dtype(self):
		return self.L.dtype	

def inner_loop(z, Y, r, weight, Hr = None):
	r"""
	Note: this assumes full data, not tangential.
	"""

	if Hr is None or Hr.state_dim != r:
		poles = 'linearized'
	else:
		poles = Hr.poles()

	# Fit 1
	vf = polyrat.VectorFittingRationalApproximation(r-1, r, poles0 = poles, verbose = False, maxiter = 10)
	vf.fit(z.reshape(-1,1), Y, weight = weight)
	lam, R = vf.pole_residue()

	output = pole_residue_to_real_statespace(lam, R, rank = 1)
	# Optimize the fit 
	H1 = fit_real_mimo_statespace_system(z, Y, *output, stable = True, weight = weight, verbose = False)
	
	if Hr is not None and Hr.state_dim == r:
		H2 = fit_real_mimo_statespace_system(z, Y, 
			Hr._alpha, Hr._beta, Hr._B, Hr._C, Hr._gamma, Hr._b, Hr._c, 
			stable = True, weight = weight, verbose = False) 

		H1_norm = _norm(weight, H1.transfer(z) - Y)
		H2_norm = _norm(weight, H2.transfer(z) - Y)

		if np.isclose(H1_norm, H2_norm):
			print("H-vf", Fore.RED, H1_norm, Style.RESET_ALL,  "H-previous", Fore.RED, H2_norm, Style.RESET_ALL)
		elif H1_norm < H2_norm:
			print("H-vf", Fore.RED, H1_norm, Style.RESET_ALL, "H-previous", H2_norm)
		else:
			print("H-vf", H1_norm, "H-previous", Fore.RED, H2_norm, Style.RESET_ALL)

		if H2_norm < H1_norm:
			return H2
		else:
			return H1 

	else:
		return H1


def score_candidates_angle0(weight, mu, mu_can):
	r""" Score the candidates based on only V[mu_can] 
	"""
	score = np.zeros(mu_can.shape)
	m = len(mu)
	for k, muc in enumerate(mu_can):
		x = (mu + muc.conj())**(-1)
		Mc = (muc + muc.conj())**(-1)
		x /= np.sqrt(Mc)
		wx = weight @ x
		score[k] = np.linalg.norm(wx)
	return score

def score_candidates_angle1(weight, mu, mu_can):
	r""" Score the candidates based on Span(V[mu_can],V'[mu_can])
	"""
	score = np.zeros(mu_can.shape)
	m = len(mu)
	nc = len(mu_can)
	X = np.zeros((m, 2), dtype = np.complex)
	Mc = np.zeros((2,2), dtype = np.complex)
	for k, muc in enumerate(mu_can):
		X[:,0] = (mu + muc.conj())**(-1)
		X[:,1] = (mu + muc.conj())**(-2)
		Mc[0,0] = (muc + muc.conj())**(-1) 
		Mc[0,1] = (muc + muc.conj())**(-2) 
		Mc[1,0] = (muc + muc.conj())**(-2)
		Mc[1,1] = 2*(muc + muc.conj())**(-3)

		R = scipy.linalg.cholesky(Mc, lower = False)
		A = weight @ scipy.linalg.solve_triangular(R, X.conj().T, lower = False, trans = 'C').conj().T
		s = scipy.linalg.svdvals(A)
		print(f"candidate {muc.real:10e} 1j{muc.imag:+10e} | singular values {s[0]:5f} {s[1]:5f}")
		s[s>1] = 1.	
		#score[k] = -np.min(np.arccos(s))
		score[k] = -np.min(s)
	return score


def project_mu(mu, mu_can, scale = 10):
	mu_can.real = np.minimum(mu_can.real, scale*np.max(mu.real))
	mu_can.real = np.maximum(mu_can.real, (1/scale)*np.min(mu.real))
	mu_can.imag = np.minimum(mu_can.imag, scale*np.max(mu.imag))
	mu_can.imag = np.maximum(mu_can.imag, scale*np.min(mu.imag))
	return mu_can


def outer_loop(H, r, mu0, maxiter = 100, score_candidates = score_candidates_angle1):
	mu = np.copy(mu0.astype(np.complex))
	Hmu = H.transfer(mu)
	Hr = None
	it = 0
	while True:
		M = Weight(mu)
		weight = M @ np.eye(Hmu.shape[0])

		Hr = inner_loop(mu, Hmu, r, weight, Hr = Hr)
		poles = Hr.poles()

		# flip poles into RHP
		mu_can = np.abs(poles.real) + 1j*poles.imag
		# Constrain growth
		mu_can = project_mu(mu, mu_can)

		err_norm = (H - Hr).norm()
		print(f"norm {err_norm:10e}; cond {M.cond:5e}")

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
				


