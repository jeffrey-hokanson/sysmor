import numpy as np
from mor import marriage_norm, marriage_sort, LagrangePolynomial


def lagrange_roots(n = 10, deflation = True):
	zhat = np.exp(2j*np.pi*np.arange(n)/n)
	zhat = 0.5*zhat + 0.5
	true_roots = np.arange(1,n)/n
	pzhat = np.array([ np.prod(z - true_roots) for z in zhat])
	p = LagrangePolynomial(zhat, pzhat)

	roots = p.roots(deflation = deflation)
	# print roots
	qzhat = np.array([ np.prod(z - roots) for z in zhat])
	# print true_roots
	I = marriage_sort(true_roots, roots)
	print("Value at roots		  ", np.linalg.norm(p(roots), np.inf))
	assert np.linalg.norm(p(roots), np.inf) < 1e-12
	print("Mismatch from true roots", np.linalg.norm(roots[I] - true_roots, np.inf))
	print("Rel. Backward error	 ", np.linalg.norm(pzhat - qzhat, np.inf)/np.linalg.norm(pzhat))

def test_lagrange_roots():
	lagrange_roots(n = 5, deflation = True)	
	lagrange_roots(n = 5, deflation = False)	

# TODO: Finite difference check of derivative


if __name__ == '__main__':
	lagrange_roots()
