import numpy as np
import mpmath as mp
from itertools import product

from functools import lru_cache

@lru_cache(maxsize = 2)
def cauchy_eigen(mu, dps):
	mu = np.atleast_1d(mu)
	n = len(mu)
	with mp.workdps(dps):
		mu = mp.matrix(mu)
		M = mp.zeros(n,n)
		for i,j in 	product(range(n), range(n)):
			M[i,j] = (mu[i] + mu[j].conjugate())**(-1) 
		ewL, QL = mp.eighe(M)
		return ewL, QL


def subspace_angle_V_M_mp(mu, lam, dps = 100):
	r"""Multiprecision implementation computing the subspace angle between V(mu) and M(lam) 
	"""
	n = len(mu)
	lam = np.atleast_1d(lam)
	m = len(lam)

	with mp.workdps(dps):
		mu = mp.matrix(mu)
		lam = mp.matrix(lam)
		
		# Construct the Cauchy mass matrix
		#M = mp.zeros(n,n)
		#for i,j in 	product(range(n), range(n)):
		#	M[i,j] = (mu[i] + mu[j].conjugate())**(-1) 
		#ewL, QL = mp.eighe(M)
		ewL, QL = cauchy_eigen(mu, dps)

		# Construct the right hand side Mhat matrix
		Mhat = mp.zeros(2*m, 2*m)
		for i,j in product(range(m), range(m)):
			Mhat[i,j]   = (-lam[i] - lam[j].conjugate())**(-1)
			Mhat[i,m+j] = (-lam[i] - lam[j].conjugate())**(-2)
			Mhat[m+i,j] = (-lam[i] - lam[j].conjugate())**(-2).conjugate()
			Mhat[m+i,m+j] = 2*(-lam[i] - lam[j].conjugate())**(-3)

		# Construct the interior matrix
		A = mp.zeros(n, 2*m)
		for i, j in product(range(n), range(m)):
			A[i,j] = (mu[i] - lam[j])**(-1)
			A[i,m+j] = (mu[i] - lam[j])**(-2)

		ewR, QR= mp.eighe(Mhat)
		
		AA = mp.diag([1/mp.sqrt(ewL[i]) for i in range(n)]) * (QL.H * A * QR) * mp.diag([1/mp.sqrt(ewR[i]) for i in range(2*m)])
		U, s, VH = mp.svd(AA)
		phi = np.array([float(mp.acos(si)) for si in s])
		return phi
		

if __name__ == '__main__':
	mu = [0.5, 1.5]
	lam = [-1]

	mu = [3.94e+01, 7.89e+01, 1.58e+02, 3.15e+02,]
	lam = [- 6.31e+02]
	for i in range(2,5):
		print(180/np.pi*subspace_angle_V_M_mp(mu[0:i], lam))	
	



