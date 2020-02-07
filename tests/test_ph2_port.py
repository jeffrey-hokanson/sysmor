import numpy as np
from mor.demos import build_iss
from mor.ph2_port import *

from mor.check_der import check_jacobian


def test_conversion():

	r = 6
	ph_sys = FitPortHamiltonianSystem(r, verbose = True)
	theta = np.random.rand(2*r+(r*(r-1))//2)	
	R, J, C = ph_sys._theta_to_RJC(theta)
	theta2 = ph_sys._RJC_to_theta(R, J, C)
	assert np.all(theta == theta2)


def test_ph2_port_fit():
	np.random.seed(1)
	H = build_iss()[0,0]

	ph2ph = ProjectedH2PortHamiltonian(4)
	ph2ph.fit(H)		

	print(ph2ph.J)
	print(ph2ph.R)
	print(ph2ph.C)

	print("relative error", (H - ph2ph).norm()/H.norm())


def test_ph2_port():
	np.random.seed(1)
	H = build_iss()

	z = 0.01 + 1j*np.linspace(-1e2, 1e2, 40)
	Hz = H[0,0].transfer(z)

	r = 4

	ph_sys = FitPortHamiltonianSystem(r)
	ph_sys.fit(z, Hz)
	print(ph_sys.J)
	print(ph_sys.R)	

	res = lambda theta: ph_sys.residual(theta, z, Hz, W = lambda x: x)
	jac = lambda theta: ph_sys.jacobian(theta, z, Hz, W = lambda x: x)
	
	R, J, C = ph_sys._init(z, Hz)

	theta = ph_sys._RJC_to_theta(R, J, C)
	theta += 0.1*np.random.randn(len(theta))
	#theta = np.random.rand(2*r+(r*(r-1))//2)	
	#theta = np.ones(2*r+(r*(r-1))//2)	
	
	err = check_jacobian(theta, res, jac)
	print("error", err)

	assert err < 1e-6



if __name__ == '__main__':
#	test_ph2_port() 
	test_ph2_port_fit() 
#	test_conversion()
