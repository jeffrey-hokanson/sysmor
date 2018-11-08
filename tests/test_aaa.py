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

def test_cleanup():
	z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
	f = np.log(2+z**4)/(1-16*z**4)

	aaa = AAARationalFit(100)
	aaa.fit(z, f)
	n1 = np.sum(aaa.Ihat)
	residual1 = np.linalg.norm(f - aaa(z))
	aaa.cleanup()
	residual2 = np.linalg.norm(f - aaa(z))
	n2 = np.sum(aaa.Ihat)

	# Check that removing doublets has not reduced solution quality
	assert np.abs(residual1 - residual2) < 1e-12

	# check that there are now fewer terms
	assert n2 < 50	


if __name__ == '__main__':
	#aaa_tanh(N = 20, m = 10)
	test_cleanup()
