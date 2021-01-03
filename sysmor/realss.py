""" Rational functions corresponding to state-space systems
"""
import numpy as np
from itertools import product


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
