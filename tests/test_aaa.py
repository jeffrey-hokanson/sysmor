import numpy as np

from mor import AAARationalFit


def aaa_tan(N = 1000, m = 10, coeff = 4, norm = np.inf):
	""" Generate residual corresponding to [Fig. 6.4, NST18]
	"""
	# Generate test data
	z = np.exp(2j*np.pi*np.linspace(0,1, N, endpoint = False))
	f = np.tan(coeff*z)
	aaa = AAARationalFit(m, verbose = False)
	aaa.fit(z, f) 
	return np.linalg.norm(aaa(z) - f, norm)

def test_tan():
	assert aaa_tan(m = 10, coeff = 4, norm = np.inf) < 1e-7
	assert aaa_tan(m = 20, coeff = 16, norm = np.inf) < 1e-6
	assert aaa_tan(m = 50, coeff = 256, norm = np.inf) < 1e-8


if __name__ == '__main__':
	aaa_tanh(N = 20, m = 10)
	#aaa_tanh()
