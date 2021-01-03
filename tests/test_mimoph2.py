import numpy as np
import sysmor
from sysmor.demos import build_iss
from sysmor.mimoph2 import inner_loop

def test_inner_loop():
	H = build_iss()
	
	z = 0.1 + 1j*np.linspace(-100,100, 100)
	Hz = H.transfer(z)

	inner_loop(z, Hz, 10)
	

if __name__ == '__main__':
	test_inner_loop()
		
