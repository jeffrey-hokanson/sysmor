import numpy as np

import sysmor
from sysmor import TFIRKA, IncrementalTFIRKA
from sysmor.tfirka import hermite_interpolant
from sysmor.demos import build_iss

def check_meier_luenberger(H, Hr, tol = 1e-6):
	# Check ML interpolation conditions
	lam = Hr.poles()
	lam = lam[np.argsort(-lam.imag)]
	Hz, Hpz = H.transfer(-lam.conj(), der = True)
	Hrz, Hrpz = Hr.transfer(-lam.conj(), der = True)

	print("Meier-Luenberger conditions")	
	for k in range(len(lam)):
		line = f'Pole {lam[k].real: 5.5e}  {lam[k].imag: 5.5e} I  |'
		Herr = np.max(np.abs(Hz[k] - Hrz[k]))
		line += f'  H err {Herr: 5.5e}  | '

		Hperr = np.max(np.abs(Hpz[k] - Hrpz[k]))
		line += f"  H' err {Hperr: 5.5e}  | "
		print(line)

	assert np.max(np.abs(Hz - Hrz)) < tol
	assert np.max(np.abs(Hpz - Hrpz)) < tol


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

def test_tfirka():
	H = build_iss()
	H = H[0,0]

	mor = TFIRKA(10)
	mor.fit(H)
	
	err = (mor - H).norm()/H.norm()
	print(err)
	
	
	check_meier_luenberger(H, mor)
	

def test_itfirka():
	H = build_iss()
	H = H[0,0]

	mor = IncrementalTFIRKA(30, ftol = 1e-12, lamtol = 1e-13, print_norm = True)
	mor.fit(H)
	
	err = (mor - H).norm()/H.norm()
	print(err)
	check_meier_luenberger(H, mor, tol = 1e-3)
	

if __name__ == '__main__':
	#test_build_Hermite()
	test_itfirka()
