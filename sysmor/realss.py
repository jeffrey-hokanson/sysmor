""" Rational functions corresponding to state-space systems
"""
import numpy as np
from itertools import product
import scipy.optimize
import scipy.linalg

from .system import StateSpaceSystem
from .marriage import hungarian_sort 
from .util import _get_dimensions

# TODO: Move into a better location
def eval_pole_residue(z, lam, R):
	M = z.shape[0]
	r, p, m = R.shape

	Y = np.zeros((M, p, m), dtype = np.complex)

	for k in range(r):
		Y += np.einsum('i,jk->ijk', 1./(lam[k] - z), R[k])

	return Y


def eval_realss(z, alpha, beta, B, C, gamma, b, c):
	M = z.shape[0]
	r_c = alpha.shape[0]
	r_r = gamma.shape[0]
	m = B.shape[1]
	p = C.shape[0]

	# TODO: better determine if we need complex storage
	Y = np.zeros((M, p, m), dtype = np.complex)
	
	# Handel the complex pairs
	Ainv = np.zeros((M,2,2), dtype = np.complex)
	for k in range(r_c):
		det = (alpha[k] - z)**2 + beta[k]**2
		Ainv[:,0,0] = (alpha[k] - z)/det
		Ainv[:,1,1] = (alpha[k] - z)/det
		Ainv[:,0,1] = -beta[k]/det
		Ainv[:,1,0] = beta[k]/det
		Y += np.einsum('jk,ikl,lm->ijm', C[:,k*2:k*2+2], Ainv, B[k*2:k*2+2,:])

	for k in range(r_r):
		#res -= np.einsum('jk,i,kl->ijl', c[:,k:k+1], 1./(gamma[k] - z), b[k:k+1,:])
		Y += np.einsum('j,i,k->ijk', c[:,k], 1./(gamma[k] - z), b[k,:])

	return Y
	

def pole_residue_to_real_statespace(lam, R, rank = None):
	r, p, m = R.shape
	if rank is None: rank = min(p,m)	

	# ensure poles come in exact conjugate pairs
	I = hungarian_sort(lam, lam.conj())
	#print("poles\n", lam)
	#print("mismatch\n", lam - lam[I].conj())
	#assert np.all(np.isclose(lam, lam[I].conj(),atol = 1e-1, rtol = 1e-1)), "Poles are not in conjugate pairs"
	lam = (lam + lam[I].conj())/2
	# Same for the residues
	#assert np.all(np.isclose(R, R[I].conj(), rtol = 1e-1)), "Residues are not in conjugate pairs"
	R = (R + R[I].conj())/2


	# Now 
	alpha = []
	beta = []
	B = []
	C = []	
	gamma = []
	b = []
	c = []
	
	T = 1./np.sqrt(2)*np.array([[1, -1j],[-1j,1]])

	r_c = 0
	r_r = 0

	idx = list(range(r))
	while idx:
		k = idx.pop()
		if k == I[k]:
			# Real pole
			r_r += 1
			U, s, VH = scipy.linalg.svd(R[k], full_matrices = False)
			for j in range(min(np.sum(~np.isclose(s,0)), rank)):
				gamma.append(lam[k].real)
				b.append(VH[j,:].real*np.sqrt(s[j]))
				c.append(U[:,j].real*np.sqrt(s[j]))
		else:
			# Complex pole
			r_c += 1
			idx.remove(I[k]) # Remove the duplicate
			
			U, s, VH = scipy.linalg.svd(R[k], full_matrices = False)
			for j in range(min(np.sum(~np.isclose(s,0)), rank)):
				alpha.append(lam[k].real)
				beta.append(lam[k].imag)
				bb = VH[j,:].T*np.sqrt(s[j])
				bb = T.conj().T @ np.vstack([bb, bb.conj()])
				# note bb/(1+j) is real
				B.append(bb.real*np.sqrt(2))
				cc = U[:,j]*np.sqrt(s[j])
				cc = np.vstack([cc, cc.conj()]).T @ T
				# note cc/(1-j) is real
				C.append(cc.real*np.sqrt(2))


	if r_c > 0:
		alpha = np.array(alpha).reshape(r_c)
		beta = np.array(beta).reshape(r_c)
		B = np.vstack(B)
		B = np.vstack(B)
		C = np.vstack([c.T for c in C]).T
	else:
		alpha = np.zeros(0)
		beta = np.zeros(0)	
		B = np.zeros((0,m))
		C = np.zeros((p,0))

	if r_r > 0:
		gamma = np.array(gamma).reshape(r_r)
		b = np.vstack(b).reshape(r_r, m)
		c = np.vstack(c).T.reshape(p, r_r)
	else:
		gamma = np.zeros(0)
		b = np.zeros((0, m))
		c = np.zeros((p, 0))

	assert B.shape == (r_c*2, m)
	assert C.shape == (p, r_c*2), f"C shape {C.shape}, expected ({p},{r_c*2})"

	return alpha, beta, B, C, gamma, b, c
	

def _residual(z, Y, alpha, beta, B, C, gamma, b, c):
	r""" Residual for a real state-space system

	Note all parameters (alpha, beta, B, C, gamma, b, c) are REAL!

	Parameters
	----------
	z: 
		(M,) array of input coordinates
	Y:
		(M, p, m) size array of output of the transfer function
	alpha: 
		(r_c,) size array containing the (1,1) and (2,2) entries of the r_c pairs of complex poles
	beta:
		(r_c) size array containing the (1,2) and (2,1) entries of the r_c pairs of complex poles
	B: 
		(r_c*2, m) size array containing the columns of B corresponding to the complex poles
	C: 
		(p, r_C*2) size array containing the columns of C corresponding
	gamma:
		(r_r,) array containing the real poles
	b:
		(r_r,m) array of columns of B corresponding to real poles
	c:
		(p,r_r) array of columns of C corresponding to real poles
	"""
	M, p, m = Y.shape
	r_c = alpha.shape[0]
	r_r = gamma.shape[0]
	# TODO: better determine if we need complex storage
	res = np.copy(Y.astype(np.complex))
	
	# Handle the complex pairs
	Ainv = np.zeros((M,2,2), dtype = np.complex)
	for k in range(r_c):
		det = (alpha[k] - z)**2 + beta[k]**2
		Ainv[:,0,0] = (alpha[k] - z)/det
		Ainv[:,1,1] = (alpha[k] - z)/det
		Ainv[:,0,1] = -beta[k]/det
		Ainv[:,1,0] = beta[k]/det
		res -= np.einsum('jk,ikl,lm->ijm', C[:,k*2:k*2+2], Ainv, B[k*2:k*2+2,:])

	for k in range(r_r):
		#res -= np.einsum('jk,i,kl->ijl', c[:,k:k+1], 1./(gamma[k] - z), b[k:k+1,:])
		res -= np.einsum('j,i,k->ijk', c[:,k], 1./(gamma[k] - z), b[k,:])
	return res


def _jacobian(z, Y, alpha, beta, B, C, gamma, b, c):
	M, p, m = Y.shape
	r_c = alpha.shape[0]
	r_r = gamma.shape[0]

	Jalpha = np.zeros((M, p, m, r_c), dtype = np.complex)
	Jbeta = np.zeros((M, p, m, r_c), dtype = np.complex)
	JB = np.zeros((M, p, m, r_c*2, m), dtype = np.complex)
	JC = np.zeros((M, p, m,  p, r_c*2), dtype = np.complex)
	Jgamma = np.zeros((M, p, m, r_r), dtype = np.complex)
	Jb = np.zeros((M, p, m, r_r, m), dtype = np.complex)
	Jc = np.zeros((M, p,m, p, r_r), dtype = np.complex)

	Ainv = np.zeros((M,2,2), dtype = np.complex)
	DAinv = np.zeros((M,2,2), dtype = np.complex)
	for k in range(r_c):
		det = (alpha[k] - z)**2 + beta[k]**2
		DAinv[:,0,0] = -(-alpha[k]-beta[k]+z)*(-alpha[k]+beta[k]+z)/det**2
		DAinv[:,1,1] = -(-alpha[k]-beta[k]+z)*(-alpha[k]+beta[k]+z)/det**2
		DAinv[:,0,1] = -(2*beta[k]*(-alpha[k]+z))/det**2
		DAinv[:,1,0] = (2*beta[k]*(-alpha[k]+z))/det**2
		Jalpha[...,k] = -np.einsum('jk,ikl,lm->ijm', C[:,k*2:k*2+2], DAinv, B[k*2:k*2+2,:])

		DAinv[:,0,0] = 2*beta[k]*(-alpha[k]+z)/det**2
		DAinv[:,1,1] = 2*beta[k]*(-alpha[k]+z)/det**2
		DAinv[:,0,1] = -(-alpha[k]-beta[k]+z)*(-alpha[k]+beta[k]+z)/det**2
		DAinv[:,1,0] = (-alpha[k]-beta[k]+z)*(-alpha[k]+beta[k]+z)/det**2
		Jbeta[...,k] = -np.einsum('jk,ikl,lm->ijm', C[:,k*2:k*2+2], DAinv, B[k*2:k*2+2,:])
	
		Ainv[:,0,0] = (alpha[k] - z)/det
		Ainv[:,1,1] = (alpha[k] - z)/det
		Ainv[:,0,1] = -beta[k]/det
		Ainv[:,1,0] = beta[k]/det

		X = -np.einsum('jk,ikl->ijl', C[:,k*2:k*2+2], Ainv)
		for j in range(m):
			JB[:,:,j,2*k,j] = X[:,:,0]
			JB[:,:,j,2*k+1,j] = X[:,:,1]
		
		X = -np.einsum('ikl,lm->ikm', Ainv, B[k*2:k*2+2,:])
		for j in range(p):
			JC[:,j,:,j,2*k] = X[:,0,:]
			JC[:,j,:,j,2*k+1] = X[:,1,:]

	for k in range(r_r):
		Jgamma[:,:,:,k] = np.einsum('jk,i,kl->ijl', c[:,k:k+1],(gamma[k] - z)**(-2), b[k:k+1,:])
		
		X = np.einsum('j,i->ij', c[:,k],(gamma[k] - z)**(-1))
		for j in range(m):	
			Jb[:,:,j,k,j] = -X[:,:]
		
		X = np.einsum('i,k->ik', (gamma[k] - z)**(-1), b[k,:])
		for j in range(p):	
			Jc[:,j,:,j,k] = -X[:,:]
	
	return Jalpha, Jbeta, JB, JC, Jgamma, Jb, Jc



def _coordinate_tangent_residual(zs, ys, alpha, beta, B, C, gamma, b, c):

	p,m = _get_dimensions(zs)
	res = []
	for i, j in product(range(p), range(m)):
		z = np.array(zs[i,j])
		y = np.array(ys[i,j]).reshape(-1,1,1)
		Bij = B[:,j].reshape(-1,1)
		Cij = C[i,:].reshape(1,-1)
		bij = b[:,j].reshape(-1,1)
		cij = c[i,:].reshape(1,-1)
		res.append(
			_residual(z, y, 
				alpha, beta, Bij, Cij, 
				gamma, bij, cij).flatten()
		)

	return np.hstack(res)

	
def _coordinate_tangent_jacobian(zs, ys, alpha, beta, B, C, gamma, b, c):
	p, m = _get_dimensions(zs)

	Jalphas = []
	Jbetas = []
	JBs = []
	JCs = []
	Jgammas = []
	Jbs = []
	Jcs = []

	for i, j in product(range(p), range(m)):
		z = np.array(zs[i,j])
		y = np.array(ys[i,j]).reshape(-1,1,1)
		Bij = B[:,j].reshape(-1,1)
		Cij = C[i,:].reshape(1,-1)
		bij = b[:,j].reshape(-1,1)
		cij = c[i,:].reshape(1,-1)
		Jalpha, Jbeta, JB, JC, Jgamma, Jb, Jc = _jacobian(
			z, y, alpha, beta, Bij, Cij, gamma, bij, cij)
	
		Jalphas.append(Jalpha.reshape(len(z), len(alpha)))
		Jbetas.append(Jbeta.reshape(len(z), len(beta)))
		
		JB_ = np.zeros((len(z), *B.shape), dtype = complex)
		JB_[:,:,j] = JB.reshape(len(z), 2*len(alpha)) 
		JBs.append(JB_)
	
		JC_ = np.zeros((len(z), *C.shape), dtype = complex)
		JC_[:,i,:] = JC.reshape(len(z), 2*len(alpha))
		JCs.append(JC_)

		Jgammas.append(Jgamma.reshape(len(z), len(gamma)))

		Jb_ = np.zeros((len(z), *b.shape), dtype = complex) 
		Jb_[:,:,j] = Jb.reshape(len(z), len(gamma))
		Jbs.append(Jb_)
		
		Jc_ = np.zeros((len(z), *c.shape), dtype = complex) 
		Jc_[:,i,:] = Jc.reshape(len(z), len(gamma))
		Jcs.append(Jc_)
	
	Jalpha = np.vstack(Jalphas)
	Jbeta = np.vstack(Jbetas)
	JB = np.vstack(JBs)
	JC = np.vstack(JCs)
	Jgamma = np.vstack(Jgammas)
	Jb = np.vstack(Jbs)
	Jc = np.vstack(Jcs)

	return Jalpha, Jbeta, JB, JC, Jgamma, Jb, Jc 
	


def _make_encoder(alpha, beta, B, C, gamma, b, c):
	def encoder(alpha, beta, B, C, gamma, b, c):
		return np.hstack([
			alpha.flatten(), 
			beta.flatten(), 
			B.flatten(),
			C.flatten(),
			gamma.flatten(),
			b.flatten(),
			c.flatten()])

	return encoder

def _make_decoder(alpha, beta, B, C, gamma, b, c):
	def decoder(x):
		it = 0
		output = []
		for mat in [alpha, beta, B, C, gamma, b, c]:
			step = int(np.prod(mat.shape))
			output.append(x[it:it+step].reshape(mat.shape))
			it += step

		return output	
	return decoder

def _make_residual(z, Y, alpha, beta, B, C, gamma, b, c, weight):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	M, p, m = Y.shape
	def residual(x):
		res = _residual(z, Y, *decoder(x))
		if weight is not None:
			for s, t in product(range(p), range(m)):
				res[slice(M),s,t] = weight @ res[:,s,t] 
		return np.hstack([res.real.flatten(), res.imag.flatten()])

	return residual

def _make_jacobian(z, Y, alpha, beta, B, C, gamma, b, c, weight):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	M, p, m = Y.shape
	def jacobian(x):
		Jacs= _jacobian(z, Y, *decoder(x))
		if weight is not None:
			for k, J in enumerate(Jacs):
				for idx in np.ndindex(J.shape[3:]):
					for s, t in product(range(p), range(m)):
						Jacs[k][(slice(M),s,t,*idx)] = weight @ J[(slice(M),s,t,*idx)] 
		J = np.hstack([	J.reshape(M*p*m,-1) for J in Jacs])
		return np.vstack([J.real, J.imag])

	return jacobian


def _make_coordinate_residual(zs, ys, alpha, beta, B, C, gamma, b, c, weights):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	p, m = _get_dimensions(zs)

	def residual(x):
		res = _coordinate_tangent_residual(zs, ys, *decoder(x))
		if weights is not None:
			start = 0
			for i, j in product(range(p), range(m)):
				length = len(zs[i,j])
				I = slice(start, start + length)
				res[I] = weights[i,j] @ res[I]
				start += length
			
		return np.hstack([res.real.flatten(), res.imag.flatten()])

	return residual


def _make_coordinate_jacobian(zs, ys, alpha, beta, B, C, gamma, b, c, weights):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	p, m = _get_dimensions(zs)

	def jacobian(x):
		Jacs = _coordinate_tangent_jacobian(zs, ys, *decoder(x))
		if weights is not None:
			start = 0
			for i, j in product(range(p), range(m)):
				length = len(zs[i,j])
				I = slice(start, start + length)
				for k, J in enumerate(Jacs):
					for idx in np.ndindex(J.shape[1:]):
						Jacs[k][(I,*idx)] = weights[i,j] @ J[(I, *idx)] 
				
				start += length

		J = np.hstack([J.reshape(J.shape[0], -1) for J in Jacs])
		return np.vstack([J.real, J.imag])

	return jacobian 


# TODO: For SIMO/MISO we can use VarPro + rational approximation parameterization

def fit_real_mimo_statespace_system(z, Y, alpha, beta, B, C, gamma, b, c, weight = None, stable =True, verbose = False):
	r"""
	"""
	encode = _make_encoder(alpha, beta, B, C, gamma, b, c)
	decode = _make_decoder(alpha, beta, B, C, gamma, b, c)
	residual = _make_residual(z, Y, alpha, beta, B, C, gamma, b, c, weight)
	jacobian = _make_jacobian(z, Y, alpha, beta, B, C, gamma, b, c, weight)


	if stable:
		bounds = (
			-np.inf, 
			encode(
				np.zeros_like(alpha),
				# Although technically we can impose beta[k]<= 0 wlog, we don't for ease of optimization
				np.inf*np.ones_like(beta), 
				np.inf*np.ones_like(B),
				np.inf*np.ones_like(C),
				np.zeros_like(gamma),
				np.inf*np.ones_like(b),
				np.inf*np.ones_like(c)
				)
			)
		# Flip to make stable
		alpha = -1*np.abs(alpha)
		gamma = -1*np.abs(gamma)
	else:
		bounds = (-np.inf, np.inf)
	
	x0 = encode(alpha, beta, B, C, gamma, b, c)

	if verbose:
		verbose = 2
	else:
		verbose = 0
	
	res = scipy.optimize.least_squares(
		residual, 
		x0,
		jac = jacobian,
		bounds = bounds,
		verbose = verbose,
		)

	return BlockStateSpaceSystem(*decode(res.x))	
	

class BlockStateSpaceSystem(StateSpaceSystem):
	r"""
	"""
	def __init__(self, alpha, beta, B, C, gamma, b, c):

		self._alpha = np.copy(alpha)
		self._beta = np.copy(beta)
		self._B = np.copy(B)
		self._C = np.copy(C)
		self._gamma = np.copy(gamma)
		self._b = np.copy(b)
		self._c = np.copy(c)

	def transfer(self, z, der = False):
		if der:
			raise NotImplementedError

		return eval_realss(z, self._alpha, self._beta, self._B, self._C, self._gamma, self._b, self._c)	

	def poles(self):
		return np.hstack([self._alpha + 1j*self._beta, self._alpha - 1j*self._beta, self._gamma	]).flatten()

	@property
	def A(self):
		n = len(self._alpha)*2+len(self._gamma)
		A = np.zeros((n,n))
		for k, (a, b) in enumerate(zip(self._alpha, self._beta)):
			A[2*k,2*k] = a
			A[2*k+1,2*k+1] = a
			A[2*k, 2*k+1] = b
			A[2*k+1, 2*k] = -b
		
		st = 2*len(self._alpha)
		for k, g in enumerate(self._gamma):
			A[st+k, st+k] = g

		return A

	@property
	def B(self):
		n = len(self._alpha)*2+len(self._gamma)
		m = self._B.shape[1]

		B = np.zeros((n,m))
		B[:2*len(self._alpha),:] = self._B
		B[2*len(self._alpha):,:] = self._b
		return B
	
	@property
	def C(self):
		n = len(self._alpha)*2+len(self._gamma)
		p = self._C.shape[0]

		C = np.zeros((p,n))
		C[:,:2*len(self._alpha)] = self._C
		C[:,2*len(self._alpha):] = self._c
		return C
