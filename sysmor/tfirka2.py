import numpy as np
import scipy
from itertools import product

from .system import DescriptorSystem


def tangential_hermite_interpolant(z, Hb, cH, cHpb, b, c):
	r"""

	From BG12, eqs. 10-12 
	"""
	
	r = len(z)
	p = len(b[0])
	m = len(c[0])
	
	A = np.zeros((r, r), dtype = complex)
	E = np.zeros((r, r), dtype = complex)

	for i, j in product(range(r), range(r)):
		if i != j:
			cHbi = cH[i] @ b[j]	# c[i] @ H[i] @ b[j]
			cHbj = c[i] @ Hb[j]	# c[i] @ H[j] @ b[j]
			A[i,j] = - (z[i]*cHbi - z[j]*cHbj)/(z[i] - z[j])
			E[i,j] = - (cHbi - cHbj)/(z[i] - z[j])
		else:
			A[i,j] = - c[i] @ Hb[i] - z[i]*cHpb[i]
			E[i,j] = - cHpb[i]

	B = np.zeros((r, p), dtype = complex)
	C = np.zeros((m, r), dtype = complex)
	
	for i in range(r):
		B[i,:] = cH[i].reshape(p)
		C[:,i] = Hb[i].reshape(m)

	return DescriptorSystem(A, B, C, E)	


def init_largest_real(H, r):
	ew, evL, evR = H.eig(which = 'LR', k = r, left = True, right = True)
	print("B", H.B.shape,"evR", evL.shape)
	B = evL.conj().T @ H.B
	print("C", H.C.shape,"evR", evR.shape)
	C = H.C @ evR
	print(B)
	print(C)
	
	#print(evL.shape)
	#print(evR.shape)
	#print("B", H.B.shape)
	#c = (H.C @ evR).T
	return ew, B, C


def tfirka(H, rom_dim, z0 = None, b0 = None, c0 = None, maxiter = 10):
	ew, b, c = init_largest_real(H, rom_dim)
	z = -ew.conj()
		
	for it in range(maxiter):
		# construct data for system
		Hb = []
		cH = []
		cHpb = []

		for i in range(rom_dim):
			Hb_, Hpb_ = H.transfer(z[i], right_tangent = b[i], der = True)
			cH_ = H.transfer(z[i], left_tangent = c[:,i])
			Hb.append(Hb_)
			cH.append(cH_)
			cHpb.append( c[:,i] @ Hpb_)
	
		# Build interpolant	
		Hr = tangential_hermite_interpolant(z, Hb, cH, cHpb, b, c.T)
		
		# Compute poles
		print("eig", Hr.eig())
		ew, b, c = init_largest_real(Hr, rom_dim)
		z = -ew.conj()	 
		print("z", z)
		assert False	
