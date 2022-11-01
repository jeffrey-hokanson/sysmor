import numpy as np
import scipy.linalg
from itertools import product

from .mimoph2 import Weight
from iterprinter import IterationPrinter
from .util import _get_dimensions
from .system import LTISystem

from polyrat import VectorFittingRationalFunction

def _least_squares(zs, ys, Ms, eta):
	p, m = _get_dimensions(zs)
	# Pre-compute QR decomposition of P 
	Ps = {}
	Qs = {}
	Rs = {}
	for i, j in zs:
		z = np.array(zs[i,j])
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
		y = np.array(ys[i,j])
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
		y = np.array(ys[i,j])
		P = Ps[i,j]

		x = Q.conj().T @ (M @ (y + np.diag(y) @ (P @ b)))
		A[i,j,:] = scipy.linalg.solve_triangular(R, x)


	return b, A	



def eval_vector_fitting(zs, A, b, eta):
	ys = {}
	for i, j in zs:
		z = np.array(zs[i,j])
		num = np.zeros(z.shape, dtype = complex)
		denom = np.ones(z.shape, dtype = complex)
		for k in range(len(eta)):
			num += A[i,j,k]/(z - eta[k])
			denom += b[k]/(z - eta[k])

		ys[i,j] = num/denom

	return ys


class VectorFittingSystem(LTISystem):
	def __init__(self, A, b, eta):
		self.A = np.copy(A).flags.writeable = False
		self.b = np.copy(b).flags.writeable = False
		self.eta = np.copy(eta).flags.writeable = False

	def transfer(self, z, der = False, left_tangent = None, right_tangent = None):
		# Mangle A to correspond to the desired system
		A = self.A
		if left_tangent is not None:
			if len(left_tangent.shape) == 1:
				left_tangent = left_tangent.reshape(1,-1)
			
			A = np.einsum('ij,jkl->ikl', left_tangent, A)

		if right_tangent is not None:
			if len(right_tangent.shape) == 1:
				right_tangent = right_tangent.reshape(-1,1)
	
			A = np.einsum('ijk,jl->ilk',A, right_tangent)

		num = np.zeros((len(Z), *A.shape[0:2]), dtype = complex)
		denom = np.ones(len(Z), dtype = complex)

		for k in range(len(self.eta)):
			num += A[:,:,k]/(z - self.eta)
			denom += self.b[k]/(z - self.eta)

		if not der:
			return num/denom

		dnum = np.zeros(num.shape, dtype = complex)
		ddenom = np.zeros(denom.shape, dtype = complex)
		
		for k in range(len(self.eta)):
			dnum -= A[:,:,k]/(z - self.eta)**2
			ddenom -= self.b[k]/(z - self.eta)**2

		Hz = num/denom

		dHz = dnum/denom - (num * ddenom)/denom**2

		return Hz, dHz
		
		 


def coordinate_vecfit(zs, ys, num_degree, denom_degree, verbose = True, Ms = None, eta0 = None,
		ftol = 1e-10, btol = 1e-7, maxiter = 100):
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


	if verbose:
		printer = IterationPrinter(it = '4d', bnorm = '10.4e', fnorm = '10.4e')
		printer.print_header(it = 'iter', bnorm = 'b norm', fnorm = 'Δ approx')

	f_old = None 

	for it in range(maxiter):	

		# Update the solution
		b, A = _least_squares(zs, ys, Ms, eta) 

		# Check termination conditions
		bnorm = np.linalg.norm(b, np.inf)
		if f_old is not None:
			f = eval_vector_fitting(zs, A, b, eta)
			fnorm = sum([np.linalg.norm(f_old[i,j] - f[i,j]) for i, j in f])
			f_old = f
		else:
			f_old = eval_vector_fitting(zs, A, b, eta)
			fnorm = None

		if verbose:
			printer.print_iter(it = it+1, bnorm = bnorm, fnorm = fnorm)
		
		if bnorm < btol or (fnorm is not None and fnorm < ftol):
			break

		# update the poles
		eta = scipy.linalg.eigvals(np.diag(eta) - np.outer(np.ones(len(eta)), b))

	return VectorFittingSystem(A, b, eta) 
