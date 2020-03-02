import numpy as np
from mor.ph2 import subspace_angle_V_M, subspace_angle_V_V
from mor import ProjectedH2MOR
from mor.demos import build_iss

def test_subspace_angle_V_M(n = 10, m = 1):

	mu = 0.1 + 1j*np.linspace(-1,1, n)
	lam = -np.arange(1,m+1)

	phi = subspace_angle_V_M(mu, lam)

	print("%12.8e %12.8e" % (phi[0], phi[1]))
	min_err = np.inf
	for h in np.logspace(-7,-1,7):
		hmu = np.hstack([ -lam + 1j*h , -lam - 1j*h ])
		phi_approx = subspace_angle_V_V(mu, hmu)
		print("%12.8e %12.8e; h= %5.2e" % (phi_approx[0], phi_approx[1], h))
		err = np.linalg.norm(phi_approx - phi, np.inf)
		if err < min_err:
			min_err = err

	assert min_err < 1e-6

def test_ph2():
	np.random.seed(0)
	ph2 = ProjectedH2MOR(2, real = True, maxiter = 10)

	H = build_iss()[0,0]

	mu0 = [1+1j, 1+2j, 1-1j, 1-2j]
	ph2.fit(H, mu0 = mu0) 


if __name__ == '__main__':
	test_subspace_angle_V_M()	
