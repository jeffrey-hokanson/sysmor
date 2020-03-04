import numpy as np
import scipy.linalg
__all__ = ['cauchy_ldl', 'cauchy_hermitian_svd']

def cauchy_ldl(mu):
	""" Compute LDL* factorization of Cauchy matrix

	Given a Hermitian Cauchy matrix specified by parameters :math:`\\boldsymbol{\mu}\in \mathbb{C}^n`
	where 

	.. math::

		\mathbf{M}(\\boldsymbol{\mu})
			= \\begin{bmatrix}
				(\mu_1 + \\bar{\mu}_1)^{-1} & \ldots &(\mu_1 + \\bar{\mu}_n)^{-1} \\\\
				\\vdots & & \\vdots \\\\
				(\mu_n + \\bar{\mu}_1)^{-1} & \ldots & (\mu_n + \\bar{\mu}_n)^{-1} 
			\end{bmatrix}

	Compute the LDL* factorization 

	.. math::

		\mathbf{M}= \mathbf{P} \mathbf{L} \mathbf{D} \mathbf{L}^* \mathbf{P}^*

	where :math:`\mathbf{L}` is a lower triangular matrix,
	:math:`\mathbf{D}` is a diagonal matrix,
	and :math:`\mathbf{P}` is a permutation matrix given by
	the permutation vector :math:`\mathbf{p}`
	
	.. math::
			
		\mathbf{P} = \mathbf{I}_{\cdot, p}.

	To compute this permutation matrix, 

	.. code::

		P = np.eye(n)[:,p]
	


	Parameters
	----------
	mu: np.ndarray (n,)
		Parameters for Cauchy matrix

	Returns
	-------
	L: np.ndarray (n,n)
		Lower triangular matrix factor
	D: np.ndarray (n,)
		entries in diagonal weighting matrix
	p: np.ndarray (n,)
		permutation vector
	"""
	n = len(mu)
	mu = np.copy(mu)
	s = 1./(mu + mu.conj())
	# Permutation vector
	p = np.arange(n)
	for k in range(n):
		jhat = k+np.argmax(np.abs(s[k:]))
		mu[[k,jhat]] = mu[[jhat,k]]
		s[[k,jhat]] = s[[jhat,k]]
		p[[k,jhat]] = p[[jhat,k]]
		
		# Update diagonal entries
		s[k+1:] = s[k+1:]*(mu[k+1:] - mu[k])*(mu[k+1:] - mu[k]).conj() / \
				( (mu[k] + mu[k+1:].conj())*(mu[k+1:] + mu[k].conj()) )

	# Now compute LDL factorization of this permuted data
	g = np.ones( (n,), dtype = mu.dtype)
	d = np.zeros((n,))
	L = np.zeros((n,n), dtype = mu.dtype)
	for k in range(n-1):
		d[k] = 2*mu[k].real #=(mu[k] + mu[k].conj())
		L[k:,k] = g[k:] / (mu[k:] + mu[k].conj())
		g[k+1:] = g[k+1:] * (mu[k+1:] - mu[k])/(mu[k+1:] + mu[k].conj())

	d[-1] = 1./(2*mu[-1].real) #1./(mu[-1] + mu[-1].conj())
	L[-1,-1] = g[-1]
	return L, d, p


def cauchy_hermitian_svd(mu, L = None, d = None, p = None):
	r""" Computes the singular value decomposition of a Hermitian Cauchy matrix
	"""

	n = len(mu)
	mu = np.array(mu)

	if (L is None) or (d is None) or (p is None):
		L, d, p = cauchy_ldl(mu)

	M = 1./(np.tile(mu.reshape(n,1), (1,n)) + np.tile(mu.conj().reshape(1,n), (n,1)))
	
	# Change to match notation in Dem00, Alg. 3 (end)
	P = np.eye(len(mu))[p]
	D = np.diag(d)
	X = L
	YH = L.conj().T

	M2 = P.T.dot(X.dot(D.dot(YH).dot(P)))

	# STEP 1: compte X*D*Pinv = Q*R
	[Q,R,p1] = scipy.linalg.qr(X.dot(D), pivoting = True, mode = 'economic')

	
	# STEP 2: W = R*P*Y'
	# We pivot the rows 
	W = np.dot(R, YH[p1,:])
	
	# STEP 3: compute svd of W
	[Ubar,s,VH] = np.linalg.svd(W, full_matrices = False, compute_uv = True)

	# STEP 4: U = Q*Ubar
	U = np.dot(Q, Ubar)

	U = P.T.dot(U)
	VH = VH.dot(P)

	return U, s, VH

