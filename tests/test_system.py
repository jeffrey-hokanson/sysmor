import numpy as np

from sysmor import *
from sysmor.demos import build_iss

import pytest

def test_pole_residue():
	np.random.seed(0)
	#H = build_iss()
	#H = H[0,0]
	A = np.diag(np.linspace(-10,-1,5)) + 1j*np.random.randn(5)
	Q,_ = np.linalg.qr(np.random.randn(*A.shape))
	B = np.ones(A.shape[0])
	C = np.ones(A.shape[0])
	H = StateSpaceSystem(Q.T.dot(A).dot(Q), B, C)
	print(H)
	lam, rho = H.pole_residue()
	H2 = PoleResidueSystem(lam, rho)
	diff = H - H2
	err = (H - H2).norm()/H.norm()
	print("error", err)
	assert err < 1e-7


def make_systems():
	np.random.seed(0)
	n = 100
	A = np.diag(np.linspace(-10,-1,n)) + 1j*np.random.randn(n)
	B = np.random.randn(n,4)
	C = np.random.randn(8,n)
	Hd = StateSpaceSystem(A, B, C)
	
	Hs = SparseStateSpaceSystem(A, B, C)
	Hiss = build_iss(sparse = True)
	return [Hs, Hd, Hiss]

@pytest.mark.parametrize("H", make_systems())
def test_transfer(H):

	N = 10
	z = np.random.randn(N) + 1j*np.random.randn(N)
	Hz, Hzp = H.transfer(z, der = True)

	# Check evaluating tangents
	left = np.random.randn(1, H.output_dim)
	right = np.random.randn(H.input_dim, 1)

	
	left_Hz, left_Hzp = H.transfer(z, left_tangent = left, der = True)
	err = left_Hz - left @ Hz
	print(err)
	assert np.allclose(err, 0)
	err = left_Hzp - left @ Hzp
	assert np.allclose(err, 0)

	
	right_Hz, right_Hzp = H.transfer(z, right_tangent = right, der = True)
	err = right_Hz -  Hz @ right
	assert np.allclose(err, 0)
	err = right_Hzp - Hzp @ right
	assert np.allclose(err, 0)

	print("checking transpose of system")
	H2 = type(H)(H.A.conj().T, H.C.conj().T, H.B.conj().T)
	H2z = H2.transfer(z.conj()).conj().transpose([0,2,1])
	err = H2z - Hz
	assert np.allclose(err, 0)

	print("checking the dervative")
	# Check the derivative
	Hz, Hpz = H.transfer(z[0], der = True)
	h = 1e-7
	# Check the derivative in several directions
	for e in [1, 1j, (1+1j)/np.sqrt(2)]:
		Hpz_est = (H.transfer(z[0] + e*h) - H.transfer(z[0] - e*h))/(2*h*np.abs(e))
		err = e*Hpz - Hpz_est
		print(err)
		assert np.allclose(err, 0, atol = 1e-6)



if __name__ == '__main__':
	test_transfer()
	
