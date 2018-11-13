import numpy as np

__all__ = ['cauchy_ldl', 'cauchy_inv_norm']

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

def cauchy_inv_norm(f, L, d, p):
	""" Evaluate the weighted 2-norm associated with an inverse Cauchy matrix
	"""	
	pass
