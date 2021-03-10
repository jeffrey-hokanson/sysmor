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
			A[i,j] = -(c[i] @ (z[i]*Hb[i] - z[j]*Hb[j])/(z[i] - z[j])
			E[i,j] = -(c[i] @ (Hb[i] - Hb[j]) )/(z[i] - z[j])
		else:
			A[i,j] = - cHpb[i]
			E[i,j] = - c[i] @ Hb[i] - z[i]*cHpb[i]

	B = np.vstack(cH)
	C = np.hstack(Hb)

	return DescriptorSystem(A, B, C, E)		
