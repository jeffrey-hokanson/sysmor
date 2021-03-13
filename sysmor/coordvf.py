import numpy as np
import scipy.linalg
from itertools import product

from .mimoph2 import Weight
from iterprinter import IterationPrinter
from .util import _get_dimensions


def _least_squares(zs, ys, Ms, eta):
	p, m = _get_dimensions(zs)
	# Pre-compute QR decomposition of P 
	Ps = {}
	Qs = {}
	Rs = {}
	for i, j in zs:
		z = zs[i,j]
		P = np.zeros((len(z), len(eta)), dtype = complex)
		for k in range(len(eta)):
			P[:,k] = 1./(z - eta[k])

		Q, R = scipy.linalg.qr(Ms[i,j] @ P, mode = 'economic')
		Ps[i,j] = P
		Qs[i,j] = Q
		Rs[i,j] = R

	# Compute b
	AA = []
	bb = []
	for i, j in zs:
		y = ys[i,j]
		M = Ms[i,j]
		Q = Qs[i,j]
		P = Ps[i,j]

		My = M @ y
		bb.append( My - Q @ (Q.conj().T @ My))
		A = M @ (np.diag(y) @ P)
		AA.append( A - Q @ (Q.conj().T @ A))

	AA = np.vstack(AA)
	bb = np.hstack(bb)

	b, _, _, s = scipy.linalg.lstsq(AA, -bb)
	# Compute a
	A = np.zeros((p,m, len(eta)), dtype = complex)
	for i, j in zs:
		R = Rs[i,j]
		Q = Qs[i,j]
		M = Ms[i,j]
		y = ys[i,j]

		x = Q.conj().T @ (M @ (y + np.diag(y) @ (P @ b)))
		A[i,j,:] = scipy.linalg.solve_triangular(R, x)


	return b, A	


def coordinate_vecfit(zs, ys, num_degree, denom_degree, verbose = True, Ms = None, eta0 = None):
	r"""


	"""
	if num_degree != denom_degree - 1:
		raise NotImplementedError


	if eta0 is None:
		z_min = np.inf
		z_max = -np.inf
		for i,j in zs:
			z_min = min(z_min, *[z.imag for z in zs[i,j]])
			z_max = max(z_max, *[z.imag for z in zs[i,j]])

		eta0 = 1j*np.linspace(z_min, z_max, denom_degree)
	
	eta = np.array(eta0)


	if Ms is None:
		Ms = {}
		for i,j in zs:
			Ms[i,j] = np.eye(len(zs[i,j]))

	for it in range(50):	
		b, A = _least_squares(zs, ys, Ms, eta) 
		# update the poles
		eta = scipy.linalg.eigvals(np.diag(eta) - np.outer(np.ones(len(eta)), b))
		print("eta", eta[np.argsort(eta.imag)])		
