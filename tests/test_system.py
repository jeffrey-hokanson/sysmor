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

	Hd = DescriptorSystem(A, B, C, np.eye(A.shape[0]))

	A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
	E = np.random.randn(n,n) + 1j*np.random.randn(n,n)
	E = E.conj().T @ E
	Hd2 = DescriptorSystem(A, B, C, E)
	return [Hs, Hd, Hiss, Hd2]

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
	if isinstance(H, DescriptorSystem):
		H2 = type(H)(H.A.conj().T, H.C.conj().T, H.B.conj().T, H.E.conj().T)
	else:
		H2 = type(H)(H.A.conj().T, H.C.conj().T, H.B.conj().T)
	H2z = H2.transfer(z.conj()).conj().transpose([0,2,1])
	err = H2z - Hz
	print(err)
	assert np.allclose(Hz, H2z)

	print("checking the dervative")
	# Check the derivative
	Hz, Hpz = H.transfer(z[0], der = True)
	h = 1e-7
	# Check the derivative in several directions
	for e in [1, 1j, (1+1j)/np.sqrt(2)]:
		Hpz_est = (H.transfer(z[0] + e*h) - H.transfer(z[0] - e*h))/(2*h*np.abs(e))
		err = e*Hpz - Hpz_est
		print(err)
		assert np.allclose(e*Hpz, Hpz_est, atol = 1e-6)


def test_diagonal():
	np.random.seed(0)
	n = 4
	m = 3
	p = 2
	
	# Check transfer evaluations
	A = np.diag(np.random.randn(n) + 1j*np.random.randn(n))
	B = np.random.randn(n,p)
	C = np.random.randn(m,n)
	H = StateSpaceSystem(A,B,C)
	
	Hd = DiagonalStateSpaceSystem(np.diag(A), B, C)
	N = 10
	z = np.random.randn(N) + 1j*np.random.randn(N)

	for right_tangent in [None, np.random.randn(p,1), np.random.randn(p)]:
		Hz = H.transfer(z, right_tangent = right_tangent)
		Hdz = Hd.transfer(z, right_tangent = right_tangent)
		print(Hz - Hdz)
		assert np.allclose(Hz, Hdz)
	
	# Test conversion to diagonal system
	A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
	
	H = StateSpaceSystem(A, B, C)
	Hd = H.to_diagonal()
	print(Hd.A)	
	Hz = H.transfer(z)
	Hdz = Hd.transfer(z)
	print(Hz[0])
	print(Hdz[0])
	assert np.allclose(Hz, Hdz)

	# Test convert from Descriptor
	print("testing conversion from DescriptorSystem")
	E = np.random.randn(n,n) +1j*np.random.randn(n,n)
	E = E.conj().T @ E
	#E = np.eye(n)
	H1 = DescriptorSystem(A, B, C, E)
	H2 = H1.to_state_space()
	H3 = H1.to_diagonal()
	print(H1.transfer(z[0]))
	print(H2.transfer(z[0]))
	assert np.allclose(H1.transfer(z), H2.transfer(z))
	assert np.allclose(H1.transfer(z), H3.transfer(z))
	
#	Hz = H.transfer(z)
#	Hdz = Hd.transfer(z)
#	print(Hz[0])
#	print(Hdz[0])
#	assert np.allclose(Hz, Hdz)

if __name__ == '__main__':
	H = build_iss(sparse = True)
	#test_transfer(H)
	test_diagonal()
	
