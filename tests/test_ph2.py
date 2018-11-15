import numpy as np
from mor import subspace_angle_V_M
from mor import ProjectedH2MOR
from mor.demos import build_string

def test_subspace_angle_V_M(n = 10, m = 1):
	#mu = np.random.uniform(0.1, 1, size = (n,) ) + 1j*np.random.randn(n)
	#lam = np.random.uniform(-1, -0.1, size = (m,)) + 1j*np.random.randn(m)

	eps = 1e-2
	mu = [1 + eps*1j, 1 - eps*1j]
	lam = [-1]

	subspace_angle_V_M(mu, lam)
	# TODO: Implement test


def test_ph2():
	ph2 = ProjectedH2MOR(10, real = True, maxiter = 10)

	H = build_string()

	mu0 = [1+1j, 1+2j, 1-1j, 1-2j]
	ph2.fit(H, mu0 = mu0) 


if __name__ == '__main__':
	test_ph2()	
