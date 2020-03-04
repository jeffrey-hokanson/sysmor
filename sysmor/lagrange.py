from __future__ import division
import numpy as np
from scipy.linalg import eig, eigvals, hessenberg
import warnings

class LagrangePolynomial:
	""" Provides access to polynomials defined in a Lagrange basis 

	This class provides access to polynomials expressed in a Lagrange basis
	given Lagrange nodes :math:`\mathbf{\widehat{z}}`: 

	.. math:: 
	
		p(z) := \sum_j a_j \phi_j(z) \quad \phi_j(z) := \prod_{k \\ne j} \\frac{ z - \widehat{z}_k}{\widehat z_j - \widehat z_k}. 

	With this construction, :math:`p(\widehat{z}_j) = a_j`.

	Parameters
	----------
	zhat : array-like
		Lagrange nodes 
	a : array-like, optional
		Values of the polynomial at the Lagrange nodes 
	"""

	def __init__(self, zhat, a = None):
		 
		# Copy over data into the class
		self.zhat = np.array(zhat).reshape(-1)
		assert len(self.zhat.shape) == 1, "Lagrange nodes must be a one-dimensional vector"
	
		if a is not None:
			self.a = np.array(np.squeeze(a))
			assert len(self.a.shape) == 1, "Values at Lagrange nodes must be scalars"
			assert self.a.shape[0] == self.zhat.shape[0], "There must be the same number of Lagrange nodes as values at those nodes"
	   	 
		# Compute weights used by barycentric form [eq. (3.2), Berrut and Trefethen 04]
		n = self.zhat.shape[0]
		self.w = 1./np.array([ np.prod(self.zhat[k] - self.zhat[0:k]) * np.prod(self.zhat[k] - self.zhat[k + 1:]) for k in range(n)])
	

	def __call__(self, z):
		""" Evaluate the polynomial at points z
	
		Given a vector of points :math:`\mathbf{z}`, evaluate the polynomial at each,
		returning the vector those values, e.g., :math:`[\mathbf{p}]_j = p(z_j)`.
		
		In order for this method to be called, the coefficients of the Lagrange polynomial :math:`\mathbf{a}`
		must have been specified on initialization.

	
		Parameters
		----------
		z: array-like
			Locations where to evaluate the polynomial
		
		Returns
		-------
		array
			Values of polynomial at z[j]
		"""
		assert self.a is not None, "Polynomial not specified; please provide values at Lagrange nodes on initialization"

		V = self.vandmat(z)
		return np.dot(V, self.a)
	
	def der(self, z):
		"""Evaluate derivative of the polynomial at points z

		This function evaluates the derivative of :math:`p` at the values in :math:`\mathbf{z}`:

		.. math::
		
			p'(z_k) = \sum_j a_j \phi_j'(z)
		
		where :math:`'` denotes derivative. This function is useful when computing the residues 
		associated with certain poles.

		In order for this method to be called, the coefficients of the Lagrange polynomial :math:`\mathbf{a}`
		must have been specified on initialization.

		Parameters
		----------
		z : array-like
			Points to evaluate the derivative of p at.

		Returns
		-------
		array-like
			The derivative at the requested points

		"""
		assert self.a is not None, "Polynomial not specified; please provide values at Lagrange nodes on initialization"

		dp = np.zeros(z.shape, dtype = np.complex)
		for k, zk in enumerate(z):
			dp[k] = self._der(zk)
		return dp

	def _der(self, z):
		""" Return derivative at a particular point
		"""
		# Because we may have to evaluate the derivative at the Lagrange nodes, we use a more complex formula following
		# https://math.stackexchange.com/questions/1105160/evaluate-derivative-of-lagrange-polynomials-at-construction-points
		pz = complex(0.0)
		n = self.zhat.shape[0]
		for j in range(n):
			pjz = complex(0.0)
			for k in range(n):
				if k != j:
					pjk = 1. / (self.zhat[j] - self.zhat[k])
					for ell in range(n):
						if ell != k and ell != j:
							pjk *= (z - self.zhat[ell]) / (self.zhat[j] - self.zhat[ell])
					pjz += pjk
			pz += pjz * self.a[j]

		#if not np.iscomplex(pz):
		#	pz = float(pz)
		return pz
	
	def vandmat(self, z):
		"""Construct the Vandermonde matrix evaluating the basis polynomials at points z

		Given a set of points :math:`\mathbf{z}`, construct the Vandermonde matrix :math:`\mathbf{V}`
		where 
 
		.. math::
	
			\mathbf{V} = \\begin{bmatrix}  
				\phi_0(z_1) & \cdots & \phi_n(z_1) \\\\
				\\vdots & 	& \\vdots  \\\\
				\phi_0(z_N) & \cdots & \phi_n(z_N) 
			\\end{bmatrix}.
 
		Here we follow Berrut and Trefethen '04, evaluating :math:`\phi_k` using the barycentric formula:

		.. math::
		
			\phi_k(z) := \\frac{w_k/(z - \widehat{z}_k)}{\sum_{j=0}^n w_j/(z - \widehat{z}_j)}.


		Parameters
		----------
		z : array-like
			Where the basis functions should be evaluated

		Returns
		-------
		array-like
			Vandermonde matrix of dimension (:code:`len(z)`, :code:`len(self.zhat)`) 
		"""
		z = np.copy(np.squeeze(z))
		z = np.atleast_1d(z)
		assert len(z.shape) == 1, "Input must be one dimensional"
	
		# Format the output type of V to be complex only if necessary 
		output_type = np.float
		if np.any(np.iscomplex(z)) or np.any(np.iscomplex(self.zhat)):
			output_type = np.complex
		output_type = np.complex
		V = np.zeros((z.shape[0], self.zhat.shape[0]), dtype=output_type)

		# The following computes the Vandermonde matrix using the barycentric formula
		# [eq. (4.2), Berrut and Trefethen '04]
		
		# The values in the denominator of the barycentric formula
		denom = np.zeros(z.shape[0], dtype=output_type)
	
		# Silence errors about divide by zero as we fix these later
		with np.errstate(divide='ignore', invalid='ignore'):
			for k in range(self.zhat.shape[0]):
				V[:, k] = self.w[k] / (z - self.zhat[k])
				denom += V[:, k]
			V = V / denom[:, None]

			# Check for exact matches (fixes NaNs and inf)
			for k in range(self.zhat.shape[0]):
				mask = z == self.zhat[k]
				V[mask, :] = 0.
				V[mask, k] = 1.
			return V



	
	def roots(self, deflation = True):
		""" Compute the roots of the polynomial

		Compute the roots :math:`\\boldsymbol{\lambda}` of the polynomial :math:`p`:

		.. math:: 
		
			p(\lambda) := \sum_j a_j \phi_j(\lambda) = 0.

		This function uses the algorithm of Cordless '07 to find these roots 
		to high-relatiev precision using a generalized eigenvalue problem:

		.. math::

			\lambda 
			\\begin{bmatrix}
				0 & \\\\
				& 1 & \\\\
				& & 1 \\\\
				& & & & \ddots \\\\
				& & & & & 1
			\\end{bmatrix}
			\mathbf{x}
			=
			\\begin{bmatrix}
				0 & -a_0 & -a_1 & \ldots & -a_n \\\\
				w_0 & \widehat{z}_0 &  & \\\\
				w_1 & & \widehat{z}_1 & &  \\\\
				\\vdots & & & \ddots & \\\\
				w_n & & & & \widehat{z}_n
			\\end{bmatrix}	
			\mathbf{x}

		Returns
		-------
		array-like
			Roots of this polynomial

		"""
		assert self.a is not None, "Polynomial not specified; please provide values at Lagrange nodes on initialization"

		# Build the LHS of the generalized eigenvalue problem
		n = self.zhat.shape[0]
		
		C1 = np.eye(n+1, dtype=np.complex)
		C1[0, 0] = 0

		# Build the RHS of the generalized eigenvalue problem
		C0 = np.zeros((n+1, n+1), dtype=np.complex)

		# a = self.a/np.linalg.norm(self.a)
		# scale = np.linalg.norm(self.w)**(1./(n-1))
		# w = self.w/scale**(n-1)
		# C0[1:n+1,1:n+1] = np.diag(self.zhat/scale)
		C0[1:n+1,1:n+1] = np.diag(self.zhat)

		# scaling
		a = self.a / np.linalg.norm(self.a)
		w = self.w / np.linalg.norm(self.w)
		# a = self.a
		# w = self.w

		C0[0,1:n+1] = a
		C0[1:n+1,0] = w
		# # balancing [LC14, eq. 29]
		a0 = np.copy(a)
		w0 = np.copy(w)
		s = np.array([1.]+[np.sqrt(np.abs(wj/aj)) if np.abs(aj) > 0 else 1 for (wj, aj) in zip(w, a)])
		C0 = np.dot(np.diag(1/s), np.dot(C0, np.diag(s)))

		# Apply a rotation to make the first weight real
		angle = np.angle(C0[1,0])
		if np.isfinite(angle):
			C0[1:,0] *= np.exp(-1j*angle)
		else:
			print("Rotation failed", angle)
			deflation = False

		if deflation:
			#C0[1,0] must be real for Householder to reflect correctly
			assert np.abs(C0[1,0].imag) < 1e-10, "C0[1,0]: %g + I %g" % (C0[1,0].real, C0[1,0].imag)
			#Householder Reflector
			u = np.copy(C0[1:,0]) # = w scaled
			u[0] += np.linalg.norm(C0[1:,0]) # (w) scaled
			H = np.eye(n, dtype=complex) - 2 * np.outer(u,u.conjugate())/(np.linalg.norm(u)**2)
			G2 = np.zeros((n+1, n+1), dtype=complex)
			G2[0,0] = 1
			G2[1:,1:] = H
			C0 = G2.dot(C0.dot(G2))
			C1 = G2.dot(C1.dot(G2))
			H1, P1 = hessenberg(C0[1:,1:], calc_q=True, overwrite_a = False)
			G3 = np.zeros((n+1, n+1), dtype=complex)
			G3[0,0] = 1
			G3[1:,1:] = P1.T.conjugate()
			G4 = np.eye(n+1, dtype=complex)
			G4[0:2,0:2] = [[0,1],[1,0]]
			H1 = G4.dot(G3.dot(C0.dot(G3.T.conjugate())))[1:,1:]
			B1 = G4.dot(G3.dot(C1.dot(G3.T.conjugate())))[1:,1:]

			# Givens Rotation
			G5 = np.eye(n, dtype=complex)
			a = H1[0,0]
			b = H1[1,0]
			c = a / np.sqrt(a**2 + b**2)
			s = b / np.sqrt(a**2 + b**2)
			G5[0:2,0:2] = [[c.conjugate(), s.conjugate()],[-s,c]]

			H2 = G5.dot(H1)[1:,1:]
			B2 = G5.dot(B1)[1:,1:]
			try:
				ew = eigvals(H2, B2)
			except np.linalg.linalg.LinAlgError as e:
				print("zhat", self.zhat)
				print("a   ", self.a)
				print(H2)
				print(B2)
				print(C0)
				raise e 
		else:
			# Compute the eigenvalues
			# As this eigenvalue problem has a double root at infinity, we ignore the division by zero warning
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message='divide by zero encountered in true_divide',
										category=RuntimeWarning)
				ew = eigvals(C0, C1, overwrite_a=False)
			ew = ew[np.isfinite(ew).flatten()]
			if ew.shape[0] != self.degree:
				print("a orig")
				print(self.a)
				print("a scaled by norm")
				print(a0)
				print("w orig")
				print(self.w)
				print("w scaled by norm")
				print(w0)
				print("s")
				print(s)
				print("lagrange nodes")
				print(self.zhat)
				assert ew.shape[0] == self.degree, "Error: too many infinite eigenvalues encountered"

		return ew

	@property
	def degree(self):
		return self.zhat.shape[0] - 1



class BarycentricPolynomial(LagrangePolynomial):
	""" A Barcentric polynomial class

	"""
	def __init__(self, zhat,  a, w):
		self.zhat = np.array(np.squeeze(zhat))
		assert len(self.zhat.shape) == 1, "Lagrange nodes must be a one-dimensional vector"
		self.a = np.array(np.squeeze(a))
		assert len(self.a.shape) == 1, "Values at Lagrange nodes must be scalars"
		self.w = np.copy(w)
		assert len(self.w.shape) == 1, "weights must be one dimensional"
		assert self.w.shape == self.a.shape == self.zhat.shape, "shapes must match"


if __name__ == '__main__':
	from marriage import marriage_norm, hungarian_sort
	n = 5
	zhat = np.exp(2j*np.pi*np.arange(n)/n)
	zhat = 0.5*zhat + 0.5
	true_roots = np.arange(1,n)/n
	pzhat = np.array([ np.prod(z - true_roots) for z in zhat])
	p = LagrangePolynomial(zhat, pzhat)

	roots = p.roots()
	# print roots
	qzhat = np.array([ np.prod(z - roots) for z in zhat])
	# print true_roots
	I = hungarian_sort(true_roots, roots)
	print("Value at roots		  ", np.linalg.norm(p(roots), np.inf))
	print("Mismatch from true roots", np.linalg.norm(roots[I] - true_roots, np.inf))
	print("Rel. Backward error	 ", np.linalg.norm(pzhat - qzhat, np.inf)/np.linalg.norm(pzhat))
	# print true_roots
	# print roots[I]
