from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular
from mor import cauchy_ldl, cholesky_inv_norm, cauchy_hermitian_svd

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
	assert err < 1e-7 
	
	diag_err = np.linalg.norm( (d - np.diag(D2))/d, np.inf)
	print "diagonal relative error", diag_err
	assert diag_err < 1e-7

def test_cholesky_inv_norm(n = 50):
	np.random.seed(1)
	mu = np.random.uniform(0.1, 1,size = (n,)) + 1j*np.random.uniform(-10,10, size = (n,))

	M = 1./(np.tile(mu.reshape(n,1), (1,n)) + np.tile(mu.conj().reshape(1,n), (n,1)))

	L, d, p = cauchy_ldl(mu)
	D = np.diag(d)
	P = np.eye(n)[:,p]

	f = np.random.randn(n) + 1j*np.random.randn(n)

	Minvf = np.linalg.solve(M, f)
	norm_M = f.conj().dot(Minvf)
	print norm_M

	norm_L = cholesky_inv_norm(f, L, d, p)
	print norm_L**2	

	assert np.abs(norm_L**2 - norm_M)/np.abs(norm_M) < 1e-7

	norm_L2 = np.linalg.norm(np.diag(d**(-0.5)).dot(solve_triangular(L, P.T.dot(f), lower = True, trans = 'N')))
	print norm_L2**2 
	assert np.abs(norm_L - norm_L2)/norm_L <1e-14

def test_cauchy_hermitian_svd(n = 10):
	np.random.seed(1)
	mu = np.random.uniform(0.1, 1,size = (n,)) + 1j*np.random.uniform(-10,10, size = (n,))

	M = 1./(np.tile(mu.reshape(n,1), (1,n)) + np.tile(mu.conj().reshape(1,n), (n,1)))

	U, s, VH = cauchy_hermitian_svd(mu)

	err = np.linalg.norm(M - U.dot(np.diag(s).dot(VH)), np.inf)
	print "error", err
	assert err < 1e-10

if __name__ == '__main__':
	test_cauchy_hermitian_svd()
