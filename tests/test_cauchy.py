from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular
from mor import cauchy_ldl

def test_cauchy_forward(n = 100):
	""" Test the error in the LDL* decomposition using
	"""
	np.random.seed(2)
	mu = np.random.uniform(-10,-0.01,size = (n,)) + 1j*np.random.uniform(-10,10, size = (n,))

	M = 1./(np.tile(mu.reshape(n,1), (1,n)) + np.tile(mu.conj().reshape(1,n), (n,1)))

	L, d, p = cauchy_ldl(mu)
	D = np.diag(d)
	
	M2 = L.dot(D.dot(L.conj().T))
	P = np.eye(n)[p]
	M2 = P.T.dot(M2.dot(P))
	err = np.linalg.norm(M - M2, np.inf)/np.linalg.norm(M, np.inf)
	assert err < 1e-15

def test_cauchy_inverse(n = 50):
	np.random.seed(1)
	mu = np.random.uniform(0.1, 1,size = (n,)) + 1j*np.random.uniform(-10,10, size = (n,))

	M = 1./(np.tile(mu.reshape(n,1), (1,n)) + np.tile(mu.conj().reshape(1,n), (n,1)))

	L, d, p = cauchy_ldl(mu)
	D = np.diag(d)
	P = np.eye(n)[:,p]

	# M = P L D L^* P^*
	# DLP = L^{-1} P.T M
	DLP = solve_triangular(L, P.T.dot(M), lower = True, trans = 'N')
	D2 = solve_triangular(L, P.T.dot(DLP.conj().T), lower = True, trans = 'N')
	D2 = D2.conj().T

	print D
	print D2
	err = np.linalg.norm(D - D2, np.inf)/np.linalg.norm(D, np.inf)
	print "Error", err
	assert err < 1e-8 
	
	diag_err = np.linalg.norm( (d - np.diag(D2))/d, np.inf)
	print "diagonal relative error", diag_err
	assert diag_err < 1e-8
if __name__ == '__main__':
	test_cauchy_inverse()
