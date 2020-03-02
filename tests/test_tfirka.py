import numpy as np

import mor
from mor import TFIRKA
from mor.tfirka import hermite_interpolant
from mor.demos import build_iss

def test_build_Hermite():
	# Test that we in-fact build a Hermite interpolant system
	H = build_iss()
	H = H[0,0]
	
	z = np.array([1+4j, 1-4j, 2+1j, 2-1j], dtype=complex)
	Hz, Hpz = H.transfer(z, der = True)

	Hr = hermite_interpolant(z, Hz.flatten(), Hpz.flatten())

	Hrz, Hrpz = Hr.transfer(z, der = True)


	err_H = np.max(np.abs(Hz - Hrz))
	err_Hp = np.max(np.abs(Hpz - Hrpz))

	#Should be numerically 0
	print("|H (z) - Hr (z)| = %1.2e" % err_H)
	print("|H'(z) - Hr'(z)| = %1.2e" % err_Hp)

	assert err_H < 1e-10
	assert err_Hp < 1e-10


if __name__ == '__main__':
	test_build_Hermite()
