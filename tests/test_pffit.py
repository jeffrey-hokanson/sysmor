import numpy as np
from scipy.optimize import least_squares
from sysmor import PartialFractionRationalFit
from sysmor.check_der import check_jacobian
from sysmor import marriage_norm, marriage_sort


def test_pf_b2lam(n = 6):
	pf = PartialFractionRationalFit(n-1, n, field = 'real')
	#np.random.seed(10)	
	#lam = np.random.randn(n) + 1j*np.random.randn(n)
	#lam = (lam + lam.conj())/2.
	
	lam = -1 + 1j*np.linspace(-2,2,n)
	b = pf._lam2b(lam)
	lam2 = pf._b2lam(b)
	I = marriage_sort(lam, lam2)
	print(lam)
	print(lam2[I])
	print(marriage_norm(lam, lam2))
 
def pf_check_jacobian_complex(z, f, W,  m = 5, n = 6):
	pf = PartialFractionRationalFit(m, n, field = 'complex')
	pf.z = z
	pf.f = f
	pf.W = W
	pf._set_scaling()
	lam = np.random.randn(n) + 1j*np.random.randn(n)
	
	res = lambda lam: pf.residual(lam.view(complex), return_real=True)
	jac = lambda lam: pf.jacobian(lam.view(complex))

	return check_jacobian(lam.view(float), res, jac)

def pf_check_jacobian_complex_plain(z, f, W,  m = 5, n = 6):
	pf = PartialFractionRationalFit(m, n, field = 'complex')
	pf.z = z
	pf.f = f
	pf.W = W
	pf._set_scaling()
	
	lam = np.random.randn(n) + 1j*np.random.randn(n)
	rho_c = np.random.randn(n+(m-n+1)) + 1j*np.random.randn(n+(m-n+1))
	x = np.hstack([lam, rho_c])
	
	res = lambda x: pf.plain_residual(x.view(complex), return_real = True)
	jac = lambda x: pf.plain_jacobian(x.view(complex))

	return check_jacobian(x.view(float), res, jac)

def pf_check_jacobian_real(z, f, W,  m = 5, n = 6):
	pf = PartialFractionRationalFit(m, n, field = 'complex')
	pf.z = z
	pf.f = f
	pf.W = W
	pf._set_scaling()
	lam = np.random.randn(n) + 1j*np.random.randn(n)
	b = pf._lam2b(lam)
	
	res = lambda b: pf.residual_real(b, return_real=True)
	jac = lambda b: pf.jacobian_real(b)

	return check_jacobian(b, res, jac)


def test_pf_jacobian_tan():
	# Generate test data
	N = 100
	coeff = 4
	z = np.exp(2j*np.pi*np.linspace(0,1, N, endpoint = False))
	f = np.tan(coeff*z)
	
	W1 = lambda x : x
	np.random.seed(0)
	W2a = np.random.randn(N,N) + 1j * np.random.randn(N,N)
	W2 = lambda x: np.dot(W2a, x)
	
	# Check for different orders, including n: even, odd and additional polynomial terms
	for (m,n) in [(2,3), (5,6), (7,6), (4,5), (6,5)]:
		# unweighted / weighted check
		for W in [W1, W2]:
			print("Checking varpro complex jacobian: m=%d, n=%d" % (m,n))
			assert pf_check_jacobian_complex(z, f, W, m, n) < 1e-7
			print("Checking varpro real jacobian: m=%d, n=%d" % (m,n))
			assert pf_check_jacobian_real(z, f, W, m, n) < 1e-7
			print("Checking plain complex jacobian: m=%d, n=%d" % (m,n))
			assert pf_check_jacobian_complex_plain(z, f, W, m, n) < 1e-7

def test_pf_fit():
	# Checks based on expected residual
	N = 100
	coeff = 4
	z = np.exp(2j*np.pi*np.linspace(0,1, N, endpoint = False))
	f = np.tan(coeff*z)
	
	pf = PartialFractionRationalFit(9,10)
	pf.fit(z, f)
	err = np.linalg.norm(f - pf(z))/np.linalg.norm(f)
	assert err <= 1e-7
	
	# Check an odd fit
	pf = PartialFractionRationalFit(10,11)
	pf.fit(z, f)
	err = np.linalg.norm(f - pf(z))/np.linalg.norm(f)
	assert err <= 1e-7

	pf = PartialFractionRationalFit(9,10, field = 'real')
	pf.fit(z, f)
	err = np.linalg.norm(f - pf(z))/np.linalg.norm(f)
	assert err <= 1e-7

def test_pf_fit_stable():
	from sysmor.demos import build_string
	string = build_string()
	z = 1j*np.linspace(-1000,1000,100)
	f = np.array([string.transfer(zz) for zz in z]).reshape(-1)
	
	for arg, kwargs in [ 
			( (9,10), {'field': 'complex'}),
			( (9,10), {'field': 'real'}),
			( (8,9), {'field': 'real'}),
		]:
		kwargs['stable'] = True
		pf = PartialFractionRationalFit(*arg, **kwargs)
		pf.fit(z, f)
		lam, rho = pf.pole_residue()
		print(kwargs)
		print(lam)
		assert np.all(lam.real <= 0), "Did not recover a stable system"

def test_pf_real():
	# Ensure the conversion into pole/residue form is accurate
	N = 100
	coeff = 4
	z = np.exp(2j*np.pi*np.linspace(0,1, N, endpoint = False))
	f = np.tan(coeff*z)
	

	for m, n in [(3,3), (3,4), (9,10), (10,11), (13,11), (12,10)]:
		print("="*10, "m,n", (m,n), "="*10)
		pf = PartialFractionRationalFit(m,n, field = 'real')
		pf.fit(z, f)
		
		#print('data', 0.5*np.linalg.norm(f)**2)
		#print("squared objective function", residual)
		r1 = f - pf(z)
		print("mismatch", r1)
		r2 = pf.residual_real(pf.b, return_real = False)
		print('residual', r2)
		assert np.linalg.norm(r1 -r2) < 1e-7
		# Check that the conversion between pole/residue and the real parameterization 
		# matches
#		converted_residual = 0.5*np.linalg.norm(f - pf(z))**2
#		assert np.abs(residual - converted_residual) < 1e-7
		
def test_pf_normalization():
	N = 100
	z = 1j*np.linspace(-10,10, N)
	z = np.random.randn(N) + 1j * np.random.randn(N)
	lam_true = np.array([ -1 -1j, -1+1j, -5+5j, -5 -5j])
	rho_true = np.array([1j, -1j, 10, 10])
	
	f = np.zeros(z.shape, dtype = np.complex) 
	for lam, rho in zip(lam_true, rho_true):
		f += rho/(z-lam)

	pf = PartialFractionRationalFit(3,4, field = 'real')
	pf.fit(z, f)
	lam, rho = pf.pole_residue()
	
	print(lam)
	print(lam_true)

	print(rho)
	print(rho_true)
		
	I = marriage_sort(lam_true, lam)
	assert np.all(np.isclose(lam_true, lam[I]))
	assert np.all(np.isclose(rho_true[I], rho[I]))

if __name__ == '__main__':
	test_pf_normalization()
