import numpy as np
import scipy.linalg as sl
import sysmor
from sysmor.varpro import *
import mpmath as mp


def test_varpro_colwise():
	np.random.seed(0)
	m = 100
	n = 10

	A = np.random.randn(m,n)
	# Make the scaling poor
	dl = np.random.permutation(np.exp(10*np.arange(m)/m))
	dr = np.random.permutation(np.exp(10*np.arange(n)/n))
	A = np.diag(dl) @ A @ np.diag(dr)
	DA = np.random.randn(m,n)
	b = np.random.randn(m,)

	a0 = np.random.randn(n)
	b = A @ a0
	#b += np.random.randn(m)

	# check only residual	
	r = varpro_residual_jacobian_colwise(b, A)

	# Check floating point
	a = sl.lstsq(A,b)[0]
	r_float = b - A @ a
	err = np.max(np.abs(r - r_float))
	print("error in residual", err)
	print("||r||/||b||", np.linalg.norm(r)/np.linalg.norm(b))	
	print("||r||/||b||", np.linalg.norm(r_float)/np.linalg.norm(b))	
	# Check extended precision
	mp.mp.dps = 50
	A_ = mp.matrix(A)
	b_ = mp.matrix(b)
	Q_, R_ = mp.qr(A_)
	r_ = b_ - Q_ * Q_.H * b_
	err = np.linalg.norm(np.array(r_, dtype = np.float) - r)
	print("error wrt MP math", err)
#	assert False

	# Check Jacobian
	r, K, L = varpro_residual_jacobian_colwise(b, A, DA)

	Q, R = sl.qr(A, mode = 'economic')
	
	K_true = DA @ np.diag(a)
	K_true -= Q @ Q.conj().T @ K_true
	K_err = np.linalg.norm(K - K_true, 'fro')
	print('K err', K_err)
	#assert np.linalg.norm(K - K_true) < 1e-6

	L_true = Q @ sl.solve(R.conj().T, np.diag(DA.conj().T @ r_true))
	L_err = np.linalg.norm(L - L_true, 'fro')
	print('L err', K_err)
	#assert np.linalg.norm(K - K_true) < 1e-6


if __name__ == '__main__':
	test_varpro_colwise()	
