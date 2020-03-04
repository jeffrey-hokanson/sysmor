# (c) Jeffrey M. Hokanson May 2018
import numpy as np
import warnings
from scipy.linalg import solve_triangular, lstsq, svd
from scipy.optimize import least_squares
from .lagrange import LagrangePolynomial, BarycentricPolynomial
from .marriage import hungarian_sort
from .ratfit import RationalFit
from .optfit import OptimizationRationalFit, BadStep
from .aaa import AAARationalFit
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


	"""
	def __init__(self, m, n, field = 'complex', stable = False, init = 'aaa', spectral_abscissa = 0, **kwargs):

		assert m + 1 >= n, "Pole-residue parameterization requires m + 1 >= n" 
		OptimizationRationalFit.__init__(self, m, n, field = field, stable = stable, init = init, **kwargs)
		if 'tr_solver' not in self.kwargs:
			self.kwargs['tr_solver'] = 'exact'
		if 'x_scale' not in self.kwargs:
			self.kwargs['x_scale'] = 'jac'
		self.spectral_abscissa = spectral_abscissa

	def _call(self, z):
#		b = self.b
#		zt = self._transform(self.z)
#		Psi = np.vstack([zt, np.ones(zt.shape)]).T	
#		zt2 = zt**2
#		Theta = [ (Psi.T/(zt2 + np.dot(Psi, b[2*k:2*k+2] ))).T for k in range(self.n//2)]
#		if self.n % 2 == 1:
#			Theta.append( 1./(zt + b[-1]).reshape(-1,1))
#	
#		if self.m - self.n >= 0:
#			Omega = np.hstack(Theta + [self._legendre_vandmat(self.m - self.n, self.z)])
# 		else:
#			Omega = np.hstack(Theta)
		
		V = self.vandmat(self.lam, z)
		return np.dot(V, self.rho_c)

	def pole_residue(self):
		return self.lam, self.rho
	
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
			print("solve_trianuglar failed")
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

		#if not np.all(np.isfinite(JRI)):
		#	raise BadStep
		
		return r.view(float), -JRI


	def residual_jacobian_real(self, b, jacobian = True, return_real = True):
		""" Construct the residual and Jacobian for the pole-residue parameterization with real pairs
		"""
	
		zt = self._transform(self.z)
		Psi = np.vstack([zt, np.ones(zt.shape)]).T	
		zt2 = zt**2
		Theta = [ (Psi.T/(zt2 + np.dot(Psi, b[2*k:2*k+2] ))).T for k in range(self.n//2)]
		if self.n % 2 == 1:
			Theta.append( 1./(zt + b[-1]).reshape(-1,1))
	
		if self.m - self.n >= 0:
			Omega = np.hstack(Theta + [self._legendre_vandmat(self.m - self.n, self.z)])
		else:
			Omega = np.hstack(Theta)

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
			if (self.n % 2 == 0) or ( k < self.n-1):
				I = [2*(k//2), 1 + 2*(k//2)]
				j = k % 2
				dOmega[:,I] = -(Psi.T * ( Psi[:,j]*(zt2 + np.dot(Psi, b[I]))**(-2))).T
			else:
				dOmega[:,k] = -1./(zt + b[-1])**2
			
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


		if self.stable:
			# Constrain the real part of the poles
			lb = -np.inf*np.ones(lam0.view(float).shape)
			ub = np.inf*np.ones(lam0.view(float).shape)
			real_part = np.ones(lam0.shape, dtype = complex).view(float) != 0.
			ub[real_part] = self.spectral_abscissa
			bounds = (lb, ub)
			lam0 = np.minimum(lam0.real, self.spectral_abscissa) + 1j*lam0.imag
		else:
			bounds = (-np.inf, np.inf)

		self._res = res = least_squares(res, lam0.view(float), jac = jac, bounds = bounds, **self.kwargs)
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
		""" Convert a set of poles into a denominator in a 2-term partial fraction expansion
		"""
		# project lam so that it is in the space of acceptable lam
		I = hungarian_sort(lam, lam.conjugate())
		lam = 0.5*(lam + lam[I].conjugate())
	
		# Transform
		lam = self._transform(lam)
		# Sort into pairs
		lam_new = [lam_i for lam_ in lam[lam.imag>0] for lam_i in [lam_, lam_.conj()] ] 
		lam_new += [lam_ for lam_ in np.sort(lam[lam.imag==0].real)]
		
		assert len(lam_new) == len(lam)
		lam = np.array(lam_new, dtype = np.complex)
		b = np.zeros(self.n)
		for i in range(self.n // 2):
			if lam[i].imag != 0:
				# If we have a complex conjugate pair, 
				# z^2 + beta*z + gamma = 0 
				beta = -2*lam[2*i].real
				delta2 = -4*lam[2*i].imag**2
				gamma = -(delta2 - beta**2)/4.
				b[2*i] = beta
				b[2*i+1] = gamma
			else:
				beta = -(lam[2*i].real + lam[2*i+1].real)
				# delta value of the discriminant
				delta = np.abs(lam[2*i].real - lam[2*i+1].real)
				gamma = -(delta**2 - beta**2)/4.
				b[2*i] = beta
				b[2*i+1] = gamma
		if (self.n % 2 == 1):
			b[-1] = -lam[-1].real
		return b

	def _b2lam(self, b):
		lam = np.zeros((self.n,), dtype = np.complex)
		for i in range(self.n // 2):
			# Here we are careful in how we compute the roots via the quadratic equation
			# to avoid cancellation.
			# https://math.stackexchange.com/q/311590
			if b[2*i] < 0:
				lam[2*i  ] = -b[2*i]/2. + np.sqrt(0j + b[2*i]**2 - 4*b[2*i+1])/2.
				lam[2*i+1] = b[2*i+1]/lam[2*i]
			elif b[2*i] > 0:
				lam[2*i  ] = -b[2*i]/2. - np.sqrt(0j + b[2*i]**2 - 4*b[2*i+1])/2.
				lam[2*i+1] = b[2*i+1]/lam[2*i]
			else:
				lam[2*i  ] = np.sqrt(0j-b[2*i+1])
				lam[2*i+1] = -np.sqrt(0j-b[2*i+1])
		
		if self.n % 2 == 1:
			lam[-1] = -b[-1]
		return self._inverse_transform(lam)


	def _fit_real(self, lam0):
		res = lambda b: self.residual_real(b, return_real = True)
		jac = lambda b: self.jacobian_real(b)
		
		# If we are enforcing a stability constraint, setup the box constraints
		

		if self.stable:
			# Push the initial poles slightly into the left half plane	
			lam0 = np.minimum(lam0.real, np.zeros(lam0.shape)) + 1j*lam0.imag
			# This will automatically symmetrize the poles
			b0 = self._lam2b(lam0)
			
			# Working through the quadratic formula,
			# [ -b +/- sqrt(b^2 - 4c) ]/2
			# has roots in the LHP if b, c are in the positive orthant
			lb = np.zeros(b0.shape)
			#lb[0::2] = -2*self.spectral_abscissa
			ub = np.inf*np.ones(b0.shape)
			bounds = (lb, ub)
			b0 = np.maximum(b0, lb)
		else:
			b0 = self._lam2b(lam0)
			bounds = (-np.inf, np.inf)
			
		# Solve the optimization problem 	
		self._res = res = least_squares(res, b0, jac, bounds = bounds, **self.kwargs)
		b = res.x
		#b, info = gn(f=res, F=jac, x0=b0, **self.kwargs)

		if self.stable:
			lam = self._b2lam(b)
			# Force into strict LHP
			lam = np.minimum(self.spectral_abscissa*np.ones(lam.shape), lam.real) + 1j * lam.imag
			b = self._lam2b(lam)
		
		# Compute residues
		r, a = self.residual_jacobian_real(b, jacobian = False, return_real = True)
		lam = self._b2lam(b)

		self.b = b
		self.lam = lam
		
		# Now compute residues
		#---------------------

		# Compute the scaling coefficient
		scale = (1./(self._transform(1) - self._transform(0))).real
		rho = np.zeros(self.n, dtype = np.complex)
		
		for i in range(self.n // 2):
			lamt1 = self._transform(lam[2*i])
			lamt2 = self._transform(lam[2*i+1])
			# This formula will produce errors if there is a double root,
			# so we catch this error and set the corresponding residues to zero
			# as the corresponding VarPro solution will have an ill-conditioned Jacobian
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				try:
					rho[2*i] = scale * (a[2*i]*lamt1 + a[2*i+1])/(lamt1 - lamt2)
					rho[2*i+1] = scale* (-a[2*i]*lamt2 - a[2*i+1])/(lamt1 - lamt2)
				except:
					pass
		
		if self.n % 2 == 1:
			rho[-1] = a[0]
		
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
