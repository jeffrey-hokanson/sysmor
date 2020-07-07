import numpy as np
from sysmor.ph2 import subspace_angle_V_M, subspace_angle_V_M_gep


def test_subspace_angle_gep():
	np.random.seed(0)
	m = 10
	r = 4
	mu = np.abs(np.random.randn(m)) + 1j*np.random.randn(m)
	lam = -np.abs(np.random.randn(r)) + 1j*np.random.randn(r)

	ang1 = subspace_angle_V_M(mu, lam)
	print(180*ang1/np.pi)
	ang2 = subspace_angle_V_M_gep(mu, lam)
	print(180*ang2/np.pi)

	assert np.max(np.abs(ang1 - ang2)) < 1e-5, "Angles not close"

if __name__ == '__main__':
	test_subspace_angle_gep()

	
