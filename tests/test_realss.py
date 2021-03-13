from itertools import product
import numpy as np
import scipy.linalg
from sysmor.realss import *
from sysmor.realss import (_residual, _jacobian, _make_encoder, _make_decoder, _make_residual, _make_jacobian,
	_coordinate_tangent_residual, _coordinate_tangent_jacobian,
	_make_coordinate_residual, _make_coordinate_jacobian)



def test_residual(p = 3, m = 2, M = 1000, r_c = 2, r_r = 3):
	np.random.seed(0)
	z = 0.1+1j*np.linspace(-1,1, M)
	Y = np.random.randn(M, p, m)
	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)

	res_true = np.zeros((M, p, m), dtype = np.complex)
	A = np.zeros((2,2), dtype = np.complex)
	for i in range(M):
		res_true[i] = Y[i]
		for k in range(r_c):
			A[0,0] = alpha[k] - z[i]
			A[1,1] = alpha[k] - z[i]
			A[0,1] = beta[k]
			A[1,0] = -beta[k]
			res_true[i] -= C[:,k*2:(k+1)*2] @ scipy.linalg.solve(A, B[k*2:(k+1)*2])

		for k in range(r_r):
			res_true[i] -= (c[:,k:k+1] @ b[k:k+1,:])/(gamma[k] - z[i])
	
	res = _residual(z, Y, alpha, beta, B, C, gamma, b, c)
	assert np.all(np.isclose(res, res_true))


def test_coord_residual(p = 3, m = 2, M = 1000, r_c = 2, r_r = 3):
	np.random.seed(0)
	z = 0.1+1j*np.linspace(-1,1, M)
	Y = np.random.randn(M, p, m)
	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)

	res_full = _residual(z, Y, alpha, beta, B, C, gamma, b, c)

	zs = {}
	ys = {}
	for i, j in product(range(p), range(m)):
		zs[i,j] = z
		ys[i,j] = Y[:,i,j]

	res_flat = _coordinate_tangent_residual(zs, ys, alpha, beta, B, C, gamma, b, c)
	err = res_flat - res_full.transpose([1,2,0]).flatten()
	assert np.allclose(res_flat, res_full.transpose([1,2,0]).flatten())
	
	Jalpha_, Jbeta_, JB_, JC_, Jgamma_, Jb_, Jc_ = _coordinate_tangent_jacobian(zs, ys, alpha, beta, B, C, gamma, b, c)

	Jalpha, Jbeta, JB, JC, Jgamma, Jb, Jc = _jacobian(z, Y, alpha, beta, B, C, gamma, b, c)
	
	assert np.allclose(Jalpha.transpose([1,2,0,3]).reshape(p*m*M,*alpha.shape), Jalpha_)
	assert np.allclose(Jbeta.transpose([1,2,0,3]).reshape(p*m*M,*beta.shape), Jbeta_)
	assert np.allclose(JB.transpose([1,2,0,3,4]).reshape(p*m*M,*B.shape), JB_)
	assert np.allclose(JC.transpose([1,2,0,3,4]).reshape(p*m*M,*C.shape), JC_)
	assert np.allclose(Jgamma.transpose([1,2,0,3]).reshape(p*m*M,*gamma.shape), Jgamma_)
	assert np.allclose(Jb.transpose([1,2,0,3,4]).reshape(p*m*M,*b.shape), Jb_)
	assert np.allclose(Jc.transpose([1,2,0,3,4]).reshape(p*m*M,*c.shape), Jc_)
	


def test_jacobian(p = 3, m = 2, M = 100, r_c = 2, r_r = 3):
	np.random.seed(0)
	z = 0.1+1j*np.linspace(-1,1, M)
	Y = np.random.randn(M, p, m)
	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)


	# Check Jalpha
	h = 1e-6
	Jalpha, Jbeta, JB, JC, Jgamma, Jb, Jc = _jacobian(z, Y, alpha, beta, B, C, gamma, b, c)

	for k in range(r_c):
		ek = np.eye(r_c)[k]
		Jest = _residual(z, Y, alpha + h*ek, beta, B, C, gamma, b, c) 
		Jest -= _residual(z, Y, alpha - h*ek, beta, B, C, gamma, b, c) 
		Jest /= 2*h
		assert np.all(np.isclose(Jalpha[...,k], Jest))

	for k in range(r_c):
		ek = np.eye(r_c)[k]
		Jest = _residual(z, Y, alpha, beta+h*ek, B, C, gamma, b, c) 
		Jest -= _residual(z, Y, alpha, beta-h*ek, B, C, gamma, b, c) 
		Jest /= 2*h
		assert np.all(np.isclose(Jbeta[...,k], Jest))
	
	for idx in np.ndindex(2*r_c, m):
		eB = np.zeros_like(B)
		eB[idx] = 1
		Jest = _residual(z, Y, alpha, beta, B + h*eB, C, gamma, b, c) 
		Jest -= _residual(z, Y, alpha, beta, B - h*eB, C, gamma, b, c) 
		Jest /= 2*h
		assert np.all(np.isclose(JB[...,idx[0], idx[1]], Jest))
	
	for idx in np.ndindex(p, 2*r_c):
		eC = np.zeros_like(C)
		eC[idx] = 1
		Jest = _residual(z, Y, alpha, beta, B, C + h*eC, gamma, b, c) 
		Jest -= _residual(z, Y, alpha, beta, B, C - h*eC, gamma, b, c) 
		Jest /= 2*h
		assert np.all(np.isclose(JC[...,idx[0], idx[1]], Jest))

	for k in range(r_r):
		ek = np.eye(r_r)[k]
		Jest = _residual(z, Y, alpha, beta, B, C, gamma+h*ek, b, c) 
		Jest -= _residual(z, Y, alpha, beta, B, C, gamma-h*ek, b, c) 
		Jest /= 2*h
		assert np.all(np.isclose(Jgamma[...,k], Jest))

	for k, j in product(range(r_r), range(m)):
		eb = np.zeros_like(b)
		eb[k,j] = 1.
		Jest = _residual(z, Y, alpha, beta, B, C, gamma, b+h*eb, c) 
		Jest -= _residual(z, Y, alpha, beta, B, C, gamma, b-h*eb, c) 
		Jest /= 2*h
		assert np.all(np.isclose(Jb[...,k,j], Jest))
	
	for k, j in product(range(r_r), range(p)):
		ec = np.zeros_like(c)
		ec[j,k] = 1.
		Jest = _residual(z, Y, alpha, beta, B, C, gamma, b, c+h*ec) 
		Jest -= _residual(z, Y, alpha, beta, B, C, gamma, b, c-h*ec) 
		Jest /= 2*h
		assert np.all(np.isclose(Jc[...,j,k], Jest))


def test_fit_real_mimo_statespace_system(p = 3, m = 2, M = 100, r_c = 2, r_r = 3):
	np.random.seed(0)
	z = 0.1+1j*np.linspace(-1,1, M)
	Y = np.random.randn(M, p, m)
	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)

	fit_real_mimo_statespace_system(z,Y, alpha, beta, B, C, gamma, b, c)


def test_encoding(p = 3, m = 2, M = 100, r_c = 2, r_r = 3):
	np.random.seed(0)
	z = 0.1+1j*np.linspace(-1,1, M)
	Y = np.random.randn(M, p, m)
	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)

	
	encoder = _make_encoder(alpha, beta, B, C, gamma, b, c)
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)

	# Check that it reproduces itself
	x = encoder(alpha, beta, B, C, gamma, b, c)
	assert np.all(x == encoder(*decoder(x)))


	W = np.random.randn(M,M)
	for weight in [None, W]:
		# Check residual 
		residual = _make_residual(z, Y, alpha, beta, B, C, gamma, b, c, weight)
		jacobian = _make_jacobian(z, Y, alpha, beta, B, C, gamma, b, c, weight)

		r = residual(x)
		J = jacobian(x)

		h = 1e-5
		for k in range(len(x)):
			ek = np.eye(len(x))[k]
			Jest = (residual(x + h*ek) - residual(x - h*ek))/(2*h)
			assert np.allclose(Jest, J[:,k])



def test_pole_residue_to_real_statespace(p = 4, m = 3):
	np.random.seed(0)
	r_c = 4
	r_r = 5

	lam = np.zeros((r_c*2+r_r), dtype = np.complex)
	R = np.zeros((r_c*2+r_r, p,m), dtype = np.complex)
	for k in range(r_c):
		lam0 = -np.random.rand() + 1j*np.random.rand()
		lam[2*k] = lam0
		lam[2*k+1] = lam0.conjugate()
		c = np.random.randn(p,1) + 1j*np.random.randn(p,1)
		b = np.random.randn(1,m) + 1j*np.random.randn(1,m)
		R[2*k] = c @ b
		R[2*k+1] = (c @ b).conj()
		
	for k in range(r_r):
		lam[2*r_c + k] = - np.random.rand()
		R[2*r_c + k] = np.random.randn(p, 1) @ np.random.randn(1,m)

	args = pole_residue_to_real_statespace(lam, R, rank = 1)

	z = 0.1+ 1j*np.linspace(-10,10,3)

	Hz1 = eval_pole_residue(z, lam, R)
	Hz2 = eval_realss(z, *args)

	print("ratio")
	print(Hz1/Hz2)	
	
	assert np.all(np.isclose(Hz1, Hz2))



def test_coordinate(p = 3, m = 2, M = 1000, r_c = 2, r_r = 3):
	np.random.seed(0)

	# Generate fake data
	zs = {(i,j):[] for i, j in product(range(p), range(m))}
	ys = {(i,j):[] for i, j in product(range(p), range(m))}
	for k in range(M):
		i = np.random.randint(p)
		j = np.random.randint(m)
		zs[i,j].append( complex(0.1 + 1j*np.random.randn(1)))
		ys[i,j].append( complex(np.random.randn(1) + 1j*np.random.randn(1))) 

	# complex conjugate poles
	alpha = -np.random.rand(r_c)
	beta = -np.random.rand(r_c)
	B = np.random.randn(r_c*2, m)
	C = np.random.randn(p, r_c*2)
	# real poles
	gamma = -np.random.rand(r_r)
	b = np.random.randn(r_r, m)
	c = np.random.randn(p, r_r)

	encoder = _make_encoder(alpha, beta, B, C, gamma, b, c)
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)


	weights = {}
	for i,j in zs:
		M = len(zs[i,j])
		weights[i,j] = np.random.randn(M,M) + 1j*np.random.randn(M,M)
	
	x = encoder(alpha, beta, B, C, gamma, b, c)
	
	for Ws in [None, weights]:
		residual = _make_coordinate_residual(zs, ys, alpha, beta, B, C, gamma, b, c, Ws)
		jacobian = _make_coordinate_jacobian(zs, ys, alpha, beta, B, C, gamma, b, c, Ws)

		r = residual(x)
		J = jacobian(x)

		h = 1e-5
		for k in range(len(x)):
			ek = np.eye(len(x))[k]
			Jest = (residual(x + h*ek) - residual(x - h*ek))/(2*h)
			err = np.max(np.abs(Jest - J[:,k]))
			print(err)
			assert np.allclose(Jest, J[:,k])
	


if __name__ == '__main__':
	#test_residual()
	#test_jacobian()
	#test_fit_real_mimo_statespace_system()
	#test_encoding()
	#test_pole_residue_to_real_statespace()
	#test_coord_residual()
	test_coordinate()
