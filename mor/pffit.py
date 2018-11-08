# (c) Jeffrey M. Hokanson May 2018
import numpy as np
from scipy.linalg import solve_triangular, lstsq, svd
from scipy.optimize import least_squares
from lagrange import LagrangePolynomial, BarycentricPolynomial
from opt.gn import gn, BadStep
from marriage import marriage_sort
from ratfit import RationalFit
from optfit import OptimizationRationalFit
from aaa import AAARationalFit
from itertools import product


class PartialFractionRationalFit(OptimizationRationalFit):
	""" Fit rational approximation using a partial fraction expansion.

	This class fits a rational approximation with :math:`m\ge n-1`
	using a partial fraction expansion. If the rational functions are posed over the complex field, 
	this class uses uses a 1-term partial fraction expansion, also known as a pole-residue parameterization:

	.. math::
		
		r(z; \\boldsymbol{\lambda}, \\boldsymbol{\\rho}, \mathbf{c}) :=
			\sum_{k=1}^n \\frac{\\rho_k}{z - \lambda_k} + \sum_{k=0}^{m-n} c_k \\varphi_k(z)

	where :math:`\\varphi_k` is polynomial basis. Then the optimization problem is posed over the real
	and imaginary parts of :math:`\\boldsymbol{\lambda}`, :math:`\\boldsymbol{\\rho}`, and :math:`\mathbf{c}`.
	When rational functions are posed over the real field, a 2-term partial fraction expansion is used 
	to impliclitly enforce this constraint

	.. math:: 

		r(z; \mathbf{a}, \mathbf{ b}, \mathbf{ c} ) = 
		\sum_{k=0}^{m-n} c_k \\varphi_k(z)
		+
		\\begin{cases}
			\displaystyle \phantom{\\frac{a_n}{z + b_n}}\phantom{+}
			\sum_{k=1}^{\lfloor n/2 \\rfloor} \\frac{ a_{2k} z + a_{2k-1}}{z^2 + b_{2k} z + b_{2k-1}}, & n \\text{ even;}\\\\
			\displaystyle \\frac{a_n}{z + b_n}
			+ \! \sum_{k=1}^{\lfloor n/2 \\rfloor} \\frac{ a_{2k} z + a_{2k-1}}{z^2 + b_{2k} z + b_{2k-1}}, 
			& n \\text{ odd;}
		\end{cases}

	and the optimization problem is posed over the *real* parameters :math:`\mathbf{a}`, :math:`\mathbf{b}`, and :math:`\mathbf{c}`.	

	Parameters
	----------
	m: int
		Degree of numerator in rational approximation
	n: int 
		Degree of denominator in rational approximation
	field: {'complex', 'real'}, optional 
		Field over which to find the rational approximation; defaults to complex
	stable: bool, optional
		If true, restrict class of rational functions to those with poles in the open left half plane; defaults to False
	init: {'aaa', 'recursive'}, optional
		Initialization technique if initial poles are not provided to the fit function
	kwargs: dict, optional
		Additional arguments to pass to the optimizer

	Attributes
	----------
	lam: array
		Poles of rational function
	rho_c: array
		Residues and coefficients of polynomial terms

	"""
	def __init__(self, m, n, field = 'complex', stable = False, init = 'aaa', **kwargs):

		assert m + 1 >= n, "Pole-residue parameterization requires m + 1 >= n" 
		OptimizationRationalFit.__init__(self, m, n, field = field, stable = stable, init = init, **kwargs)
	

	def __call__(self, z):
		V = self.vandmat(self.lam, z)
		return np.dot(V, self.rho_c)

	
	def vandmat(self, lam, z = None):
		""" Builds the Vandermonde-like matrix for the pole-residue parameterization
		""" 
		if z is None:
			z = self.z

		# Compute terms like  (z - lam)^{-1}
		V = 1./(np.tile(z.reshape(-1,1), (1,self.n)) -  np.tile(lam.reshape(1,-1), (len(z), 1)))

		# Add a polynomial of degree self.m - self.n, which has self.m - self.n + 1 columns
		# Here for numerical stability we use a Legendre polynomial basis scaled for the data
		if self.m - self.n >= 0:
			V = np.hstack([V, self._legendre_vandmat(self.m - self.n , z)])

		return V

	def vandmat_der(self, lam):
		""" Builds the column-wise derivative of the Vandermonde-like matrix for the pole-residue parameterization
		""" 
		Vp = (np.tile(self.z.reshape(-1,1), (1,self.n)) -  np.tile(lam.reshape(1,-1), (len(self.z), 1)))**(-2)
		return Vp
	
	def residual(self, lam, return_real = False):
		return self.residual_jacobian(lam, return_real = return_real, jacobian = False)

	def jacobian(self, lam):
		r, J = self.residual_jacobian(lam, return_real = False, jacobian = True)
		return J

	def residual_jacobian(self, lam, return_real = True, jacobian = True):
		
		V = self.vandmat(lam)

		# Apply mass matrix to V
		WV = self.W(V)
		Wf = self.W(self.f)
		
		# Compute the short-form QR factorization of WV to solve system
		WV_Q, WV_R = np.linalg.qr(WV, mode='reduced')
		b = np.dot(WV_Q.conjugate().T, Wf)
		try:
			a = solve_triangular(WV_R, b)
		except Exception as e:
			print "solve_trianuglar failed"
			raise e

		# residual
		PWf = np.dot(WV_Q, b)
		r = Wf - PWf
		
		if jacobian is False:
			if return_real: return r.view(float)
			else: return r
		
		DV = self.vandmat_der(lam)
		WDV = self.W(DV)

		# Compute the Jacobian
		# Storage for Jacobian in real/imaginary format
		JRI = np.empty((r.shape[0] * 2 , (lam.shape[0])* 2), dtype=np.float)

		# First term in Jacobian
		WDVa = WDV * a[:self.n] # This is equivalent to np.dot(DV, np.diag(a))
		K = WDVa - np.dot(WV_Q, np.dot(WV_Q.conjugate().T, WDVa))
		JRI[0::2, 0::2] = K.real
		JRI[1::2, 1::2] = K.real
		JRI[0::2, 1::2] = -K.imag
		JRI[1::2, 0::2] = K.imag

		# Second term in the Jacobian
		L = np.zeros((self.m + 1,), dtype = np.complex)
		L[:self.n] = np.dot(WDV.conjugate().T, r)
		L = np.dot(WV_Q, solve_triangular(WV_R, np.diag(L), trans = 'C'))
		L = L[:,:self.n]

		JRI[0::2, 0::2] += L.real
		JRI[1::2, 1::2] += -L.real
		JRI[0::2, 1::2] += L.imag
		JRI[1::2, 0::2] += L.imag

		if not np.all(np.isfinite(JRI)):
			raise BadStep
		
		return r.view(float), -JRI


	def residual_jacobian_real(self, b, jacobian = True, return_real = True):
		""" Construct the residual and Jacobian for the pole-residue parameterization with real pairs
		"""
	
		zt = self._transform(self.z)
		Psi = np.vstack([zt, np.ones(zt.shape)]).T	
		zt2 = zt**2
		if self.n % 2 == 0:
			Omega = np.hstack([ (Psi.T/(zt2 + np.dot(Psi, b[2*k:2*k+2] ))).T for k in range(self.n//2)])
		else:	
			Omega = 1./(zt + b[0]).reshape(-1,1)
			if self.n > 1:
				Omega1 = np.hstack([ (Psi.T/(zt2 + np.dot(Psi, b[2*k+1:2*k+3] ))).T for k in range(self.n//2)])
				Omega = np.hstack([Omega, Omega1])
	
		if self.m - self.n >= 0:
			Omega = np.hstack([Omega, self._legendre_vandmat(self.m - self.n, self.z)])
 
		WOmega = self.W(Omega)
		Wf = self.W(self.f)

		# Now make into real/imaginary form
		WOmegaRI = np.zeros((WOmega.shape[0]*2, WOmega.shape[1]), dtype = np.float)
		WOmegaRI[0::2,:] = WOmega.real
		WOmegaRI[1::2,:] = WOmega.imag
		WfRI = Wf.view(float)

		# Compute the short-form QR factorization of WV to solve system
		WOmegaRI_Q, WOmegaRI_R = np.linalg.qr(WOmegaRI, mode='reduced')
		c = np.dot(WOmegaRI_Q.T, WfRI)
		
		# First we compute the coefficients for the numerator polynomial
		a = solve_triangular(WOmegaRI_R, c) 
		
		# Compute the residual		
		rRI = WfRI - np.dot(WOmegaRI, a) 
		
		# Stop if we don't need to compute the jacobian
		if jacobian is False:
			if return_real: return (rRI, a)
			else: return (rRI.view(complex), a)

		# Now construct the Jacobian
		JRI = np.empty( (self.z.shape[0] * 2, self.n), dtype = np.float64)

		for k in range(self.n):
			dOmega = np.zeros(Omega.shape, dtype = np.complex)
			if self.n % 2 == 1 and k == 0:
				dOmega[:,0] = -1./(zt + b[0])**2
			elif self.n % 2 == 1:
				I = [1 + 2*((k-1)//2), 2 + 2*((k-1)//2)]
				j = (k+1) % 2
				dOmega[:,I] = -(Psi.T * ( Psi[:,j]*(zt2 + np.dot(Psi, b[I]))**(-2))).T
			elif self.n % 2 == 0:
				I = [2*(k//2), 1 + 2*(k//2)]
				j = k % 2
				dOmega[:,I] = -(Psi.T * ( Psi[:,j]*(zt2 + np.dot(Psi, b[I]))**(-2))).T
			else:
				raise NotImplementedError
			
			dWOmega = self.W(dOmega)
			
			# Now form the real imaginary version
			dWOmegaRI = np.zeros((dWOmega.shape[0]*2, dWOmega.shape[1]), dtype = np.float)
			dWOmegaRI[0::2,:] = dWOmega.real
			dWOmegaRI[1::2,:] = dWOmega.imag

			# Compute the first term in the VARPRO Jacobian
			dWOmegaRI_a = np.dot(dWOmegaRI, a)
			L = -(dWOmegaRI_a - np.dot(WOmegaRI_Q, np.dot(WOmegaRI_Q.conj().T, dWOmegaRI_a)))
			
			JRI[:, k] = L

			# Compute the second term in the VARPRO Jacobian
			dWOmegaRI_r = np.dot(dWOmegaRI.T, rRI)
			K = -np.dot(WOmegaRI_Q, solve_triangular(WOmegaRI_R, dWOmegaRI_r, trans = 'T'))

			JRI[:,k] += K
		
		return (rRI, JRI, a)	


	def residual_real(self, b, return_real = False):
		return self.residual_jacobian_real(b, jacobian = False, return_real = return_real)[0]

	def jacobian_real(self, b):
		r, JRI = self.residual_jacobian_real(b, jacobian = True)[0:2]
		return JRI

	def _fit(self, lam0):

		if self.field == 'real':
			return self._fit_real(lam0)
		else:
			return self._fit_complex(lam0)

	def _fit_complex(self, lam0):
		res = lambda lam: self.residual(lam.view(complex), return_real=True)
		jac = lambda lam: self.jacobian(lam.view(complex))

		#if self.trajectory == 'polynomial':
		#	trajectory = lambda x0, p, t: self.polynomial_trajectory(x0.view(complex), p.view(complex), t).view(float)
		#elif self.trajectory == 'pole':
		#	trajectory = lambda x0, p, t: self.pole_trajectory(x0.view(complex), p.view(complex), t).view(float)

		#lamRI, info = gn(f=res, F=jac, x0=lam0.view(float), trajectory = trajectory, **self.kwargs)
		#lam = lamRI.view(complex)
		#self.lam = lam

		res = least_squares(res, lam0.view(float), jac = jac, **self.kwargs)
		lam = res.x.view(complex)
		self.lam = lam

		# Compute residues and additional polynomial terms
		V = self.vandmat(lam)
		WV = self.W(V)
		Wf = self.W(self.f)

		rho_c = lstsq(WV, Wf, lapack_driver = 'gelss')[0] 
		self.rho_c = rho_c
		rho = rho_c[:self.n]
		c = rho_c[self.n:]
		self.rho = rho
		self.c = c


	def _lam2b(self, lam):
		# project lam so that it is in the space of acceptable lam
		I = marriage_sort(lam, lam.conjugate())
		lam = 0.5*(lam + lam[I].conjugate())
		
		# Transform
		lam = self._transform(lam)

		# Split into part of complex pairs and real eigenvalues
		lam_imag = lam[lam.imag > 0]
		lam_real = lam[lam.imag == 0].real
		lam_real = np.sort(lam_real)

		# setup initial b
		b = np.zeros(self.n)
		j_real = 0
		j_imag = 0
		i = 0
		if self.n % 2 == 1:
			b[0] = -lam_real[0]
			j_real += 1
			i += 1

		while i < self.n:
			if j_real < len(lam_real):
				# Compute the coefficients for the quadratic polynomial 
				# z^2 + beta * z + gamma
				# From the quadratic formula, the roots are -beta/2 +/- sqrt(beta^2 - 4\gamma)/2
				# beta = -(lam0 + lam1)
				beta = -(lam_real[j_real] + lam_real[j_real+1])
				# delta value of the discriminant
				delta = np.abs(lam_real[j_real] - lam_real[j_real+1])
				gamma = -(delta**2 - beta**2)/4.
				b[i] = beta
				b[i+1] = gamma
				j_real += 2
			else:
				beta = -2*lam_imag[j_imag].real
				delta2 = -4*lam_imag[j_imag].imag**2
				gamma = -(delta2 - beta**2)/4.
				b[i] = beta
				b[i+1] = gamma
				j_imag += 1	
			i += 2
		return b

	def _b2lam(self, b):
		lam = []
		i = 0
		if self.n % 2 == 1:
			lam.append(-b[i])
			i+=1
		while i < self.n:
			lam.append( -b[i]/2. + np.sqrt(0j + b[i]**2 - 4*b[i+1])/2.)
			lam.append( -b[i]/2. - np.sqrt(0j + b[i]**2 - 4*b[i+1])/2.)
			i += 2


		lam = np.array(lam, dtype = np.complex)
		return self._inverse_transform(lam)


	def _fit_real(self, lam0):
		b0 = self._lam2b(lam0)
		res = lambda b: self.residual_real(b, return_real = True)
		jac = lambda b: self.jacobian_real(b)
		
		b, info = gn(f=res, F=jac, x0=b0, **self.kwargs)
		# Compute residues
		r, a = self.residual_jacobian_real(b, jacobian = False, return_real = True)
		lam = self._b2lam(b) 
		self.b = b
		self.lam = lam
		rho = np.zeros(self.n, dtype = np.complex)
		# Now compute residues
		i = 0
		if self.n % 2 == 1:
			rho[i] = a[0]
			i += 1
		while i < self.n:
			lamt = self._transform(lam[i])
			# Compute derivative for chain rule use
			# since linear, we can simply use this formula
			scale = (1./(self._transform(1) - self._transform(0))).real
			rho[i] = scale * (a[i]*lamt + a[i+1])/(2*lamt + b[i])
			lamt = self._transform(lam[i+1])
			rho[i+1] = scale * (a[i]*lamt + a[i+1])/(2*lamt + b[i])
			i += 2

		self.b = b
		self.rho = rho
		self.c = a[self.n:]
		self.rho_c = np.hstack([self.rho, self.c])

	def plain_residual(self, lam, return_real = False):
		return self.plain_residual_jacobian(lam, return_real = return_real, jacobian = False)

	def plain_jacobian(self, lam):
		r, J = self.plain_residual_jacobian(lam, return_real = False, jacobian = True)
		return J

	def plain_residual_jacobian(self, x, return_real = False, jacobian = True):
		assert self.field == 'complex', "plain Jacobian only implemented for the complex field"
		lam = x[:self.n]
		rho_c = x[self.n:]
		
		V = self.vandmat(lam)
		# Apply mass matrix to V
		WV = self.W(V)
		Wf = self.W(self.f)

		# Compute residual
		r = Wf - np.dot(WV, rho_c)
		if not jacobian:
			if return_real: return r.view(float)
			else: return r
		
		# Compute Jacobian
		DV = self.vandmat_der(lam)
		WDV = self.W(DV)
	
		# Storage for Jacobian in real/imaginary format
		JRI = np.empty((r.shape[0] * 2 , 2*(self.n + self.m+1)), dtype=np.float)

		Llam = np.dot(WDV, np.diag(rho_c[:self.n]))
		L = np.hstack([Llam, WV])
		
		JRI[0::2, 0::2] = L.real 
		JRI[0::2, 1::2] = -L.imag
		JRI[1::2, 0::2] = L.imag 
		JRI[1::2, 1::2] = L.real 
		
		return r.view(float), -JRI	
		
if __name__ == '__main__':
	N = 100
	coeff = 4
	z = np.exp(2j*np.pi*np.linspace(0,1, N, endpoint = False))
	f = np.tan(coeff*z)
	
	pf = PartialFractionRationalFit(9,10, field = 'real')
	pf.fit(z, f)
	print("Error %5.3e" % (np.linalg.norm(f - pf(z))/np.linalg.norm(f),))
