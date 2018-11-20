import numpy as np

from mor import PoleResidueSystem, StateSpaceSystem
from mor.demos import build_iss

def test_pole_residue():
	#H = build_iss()
	#H = H[0,0]
	A = np.diag(np.linspace(-10,-1,5)) + 1j*np.random.randn(5)
	Q,_ = np.linalg.qr(np.random.randn(*A.shape))
	B = np.ones(A.shape[0])
	C = np.ones(A.shape[0])
	H = StateSpaceSystem(Q.T.dot(A).dot(Q), B, C)

	lam, rho = H.pole_residue()
	H2 = PoleResidueSystem(lam, rho)

	err = (H - H2).norm()/H.norm()
	print "error", err
	assert err < 1e-7


