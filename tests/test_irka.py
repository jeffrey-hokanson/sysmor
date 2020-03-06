import numpy as np

import sysmor
from sysmor import IRKA
from sysmor.demos import build_iss


def test_irka():
	H = build_iss()
	H = H[0,0]

	mor = IRKA(10, ftol = 1e-12)
	mor.fit(H)
	
	err = (mor - H).norm()/H.norm()
	print(err)


	# Check ML interpolation conditions
	lam = mor.poles()
	Hz, Hpz = H.transfer(-lam.conj(), der = True)
	Hrz, Hrpz = mor.transfer(-lam.conj(), der = True)
	
	for k in range(len(lam)):
		print(lam[k], 
			np.max(np.abs(Hz[k] - Hrz[k])), 
			np.max(np.abs(Hpz[k] - Hrpz[k]))
		) 

	assert np.max(np.abs(Hz - Hrz)) < 1e-8
	assert np.max(np.abs(Hpz - Hrpz)) < 1e-8

if __name__ == '__main__':
	test_irka()
