import numpy as np
import scipy
from itertools import product

from .system import DescriptorSystem, DiagonalStateSpaceSystem


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



def modal_truncation(H, r, which = 'LR'):
	ew, evR = H.eig(which = which, k = r, right = True)
	C = H.C @ evR
	B = scipy.linalg.lstsq(evR, H.B)[0]
	return DiagonalStateSpaceSystem(ew, B, C)

def tfirka(H, rom_dim, Hr0 = None, maxiter = 10, verbose = True):
	if Hr0 is None:
		Hr = modal_truncation(H, rom_dim)
	
	for it in range(maxiter):
		Hr = Hr.to_diagonal()
		
		z = -Hr.ew.conj()
		# construct data for system
		Hb = []
		cH = []
		cHpb = []
		for i in range(rom_dim):
			Hb_, Hpb_ = H.transfer(z[i], right_tangent = Hr.B[i], der = True)
			cH_ = H.transfer(z[i], left_tangent = Hr.C[:,i])
			Hb.append(Hb_)
			cH.append(cH_)
			cHpb.append( Hr.C[:,i] @ Hpb_)
	
		# Build interpolant	
		Hr = tangential_hermite_interpolant(z, Hb, cH, cHpb, Hr.B, Hr.C.T)
		Hr = Hr.to_diagonal()
		
		print("z", z[np.argsort(z.imag)])
