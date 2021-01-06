import numpy as np
import scipy.linalg
import sysmor
from sysmor.demos import build_iss
from sysmor.mimoph2 import *
from sysmor.cauchy import cauchy_ldl


def test_inner_loop():
	H = build_iss()
	
	z = 0.1 + 1j*np.linspace(-100,100, 100)
	Hz = H.transfer(z)

	inner_loop(z, Hz, 10)


def test_weight():
	m = 20
	mu = 1 + 1j*np.linspace(-10,10,m)
	#X = np.random.randn(m, 4)
	X = np.eye(m)

	L, d, p = cauchy_ldl(mu)
	Linv_true = np.diag(d**(-0.5)) @ scipy.linalg.solve(L, X[p])
	
	weight = Weight(mu)
	Linv = weight @ X
	assert np.all(np.isclose(Linv, Linv_true))

	C = 1./(np.tile(mu.reshape(m,1), (1,m)) + np.tile(mu.conj().reshape(1,m), (m,1)))	
	Csqrt = scipy.linalg.sqrtm(C)
	# If we apply Linv to Csqrt, we should have a matrix whose columns sum to one 
	U = Linv @ Csqrt
	y = np.sum(np.abs(U)**2, axis = 0)
	assert np.all(np.isclose(y, 1))


def test_outer_loop():	
	H = build_iss()
	r = 10
	mu0 = 0.01 + 1j*np.linspace(-100,100, 2*r+30)

	outer_loop(H, r, mu0)	

if __name__ == '__main__':
	#test_inner_loop()
	#test_weight()	
	test_outer_loop()
