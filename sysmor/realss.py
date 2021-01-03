""" Rational functions corresponding to state-space systems
"""
import numpy as np
from itertools import product
import scipy.optimize
import scipy.linalg

from .system import StateSpaceSystem
from .marriage import hungarian_sort 

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
	assert np.all(np.isclose(lam, lam[I].conj())), "Poles are not in conjugate pairs"
	lam = (lam + lam[I].conj())/2
	# Same for the residues
	assert np.all(np.isclose(R, R[I].conj())), "Residues are not in conjugate pairs"
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
				gamma.append(lam[k])
				print(VH[j,:])
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
	
	# Handel the complex pairs
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

def _make_residual(z, Y, alpha, beta, B, C, gamma, b, c):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	def residual(x):
		res = _residual(z, Y, *decoder(x))
		return np.hstack([res.real.flatten(), res.imag.flatten()])

	return residual

def _make_jacobian(z, Y, alpha, beta, B, C, gamma, b, c):
	decoder = _make_decoder(alpha, beta, B, C, gamma, b, c)
	M, m, p = Y.shape
	def jacobian(x):
		Jacs= _jacobian(z, Y, *decoder(x))
		J = np.hstack([	J.reshape(M*p*m,-1) for J in Jacs])
		return np.vstack([J.real, J.imag])

	return jacobian

# TODO: For SIMO/MISO we can use VarPro + rational approximation parameterization

def fit_real_mimo_statespace_system(z, Y, alpha, beta, B, C, gamma, b, c, weight = None, stable =True):
	r"""
	"""
	encode = _make_encoder(alpha, beta, B, C, gamma, b, c)
	decode = _make_decoder(alpha, beta, B, C, gamma, b, c)
	residual = _make_residual(z, Y, alpha, beta, B, C, gamma, b, c)
	jacobian = _make_jacobian(z, Y, alpha, beta, B, C, gamma, b, c)

	x0 = encode(alpha, beta, B, C, gamma, b, c)

	if stable:
		bounds = (
			-np.inf*np.ones_like(x0), 
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
	else:
		bounds = (-np.inf, np.inf)
	
	res = scipy.optimize.least_squares(
		residual, 
		x0,
		jac = jacobian,
		bounds = bounds,
		verbose = 2,
		)

	return 	
	

class RealMIMOStateSpaceSystem(StateSpaceSystem):
	r"""
	"""
	pass
