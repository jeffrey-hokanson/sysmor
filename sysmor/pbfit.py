import numpy as np
from scipy.linalg import solve_triangular, lstsq, svd
#from opt.gn import gn, BadStep
from .ratfit import RationalFit
from .aaa import AAARationalFit

from .check_der import check_jacobian, check_gradient

class PolynomialBasisRationalFit(RationalFit):
	"""Fits a rational approximation parameterized using two polynomial bases

	Given a basis for the numerator :math:`\lbrace \phi_k\\rbrace_{k=0}^m`
	and denominator :math:`\lbrace \psi_k\\rbrace_{k=0}^n`, this class fits a rational
	approximation to :math:`z_j, f(z_j)` using the paramterization

	.. math::

		r(z; \mathbf{a}, \mathbf{b}) := 
			\\frac{\sum_{k=0}^m a_k \phi_k(z)}{\sum_{k=0}^n b_k \psi_k(z)}.

	This class offers two choices of bases for the numerator and denominator separately:
	
	- A **Legendre polynomial** that has been scaled based on :math:`z` to improve conditioning
	- A **Lagrange polynomial** specified in terms of provided interpolation points

	Parameters
	----------
	m : int
		degree of polynomial in numerator
	n : int
		degree of polynomial in denominator
	field: {'complex', 'real'}, optional 
		Field over which to find the rational approximation; defaults to complex
	init: {'aaa', 'recursive'}, optional
		Initialization technique if initial poles are not provided to the fit function
	numerator_basis: {'legendre', 'lagrange'}, optional
		Basis for the numerator polynomial; defaults to legendre
	denominator_basis: {'legendre', 'lagrange'}, optional
		Basis for the denominator polynomial; defaults to legendre
	zhat_numerator: array-like, optional
		The :math:`m+1` Lagrange nodes specifying the Lagrange basis for the numerator
	zhat_denominator: array-like, optional
		The :math:`n+1` Lagrange nodes specifying the Lagrange basis for the denominator
	normalize: {'monic', 'norm'}, optional
		Normalization for the free parameter the parameterization.  
		If :code:`monic`, the denominator polynomial is monic.
		If :code:`norm`, the coefficients in the denominator polynomial have unit 2-norm. 
	kwargs: dict, optional
		Additional arguments to pass to the optimizer
	"""

	def __init__(self, m, n, field = 'complex', init = 'aaa', numerator_basis = 'legendre', denominator_basis = 'legendre',
		zhat_numerator = None, zhat_denominator = None,	normalize = 'monic', **kwargs):
		assert m >= 0, "polynomial degree of the numerator must be non negative"
		assert n >= 0, "polynomial degree of the denominator must be non negative"
		self.m = m
		self.n = n

		self.W = W
		self.real = real

		if zhat_numerator is not None:
			numerator_basis = 'lagrange'
		if zhat_denominator is not None:
			denominator_basis = 'lagrange'

		assert numerator_basis in ['legendre', 'lagrange'], "Invalid basis for numerator"
		assert denominator_basis in ['legendre', 'lagrange'], "Invalid basis for denominator"

		self.numerator_basis = numerator_basis
		self.denominator_basis = denominator_basis
		self.kwargs = kwargs
	
		self.zhat_numerator = zhat_numerator
		self.zhat_denominator = zhat_denominator

		assert init in ['aaa'], "Value of init not recogninzed"
	
		if init == 'aaa':
			self._init = self._init_aaa

		assert normalize in ['monic', 'norm']

		self.normalize = normalize

	def _convert_lam0(self, lam0):
		# Convert lam0 into space of b
		if self.denominator_basis == 'legendre':
			b0 = np.polynomial.legendre.legfromroots(self._transform(lam0))
		else:
			Psi = self._Psi
			V = np.tile( self.z.reshape(-1,1), (1, len(lam0)) ) - np.tile(lam0.reshape(1,-1), (len(self.z.shape), 1))
			h = np.prod(V, axis = 1)
			b0 = np.linalg.lstsq(Psi, h, rcond = -1)[0]
			b0 /= np.linalg.norm(b0)
		return b0

	def _fit(self, lam0):

		# Setup Lagrange nodes if these have not been specified
		if self.numerator_basis == 'lagrange' and self.zhat_numerator is None:
			self.zhat_numerator = self._generate_zhat(self.m)
		if self.denominator_basis == 'lagrange' and self.zhat_denominator is None:
			self.zhat_denominator = self._generate_zhat(self.n)

		self._Phi = self.numerator_vandmat()
		self._Psi = self.denominator_vandmat()

		if lam0 is None:
			lam0 = self._init() 

		b0 = self._convert_lam0(lam0)
		# Setup the residual and jacobian
		if self.real:
			b0 = b0.real
			res = lambda b: self.residual(b, return_real = True)
			jac = lambda b: self.jacobian(b)

			if self.normalize == 'monic':
				b0 = b0[0:-1]/b0[-1]
				b, info = gn(f=res, F=jac, x0=b0, **self.kwargs)
				self.b = b = np.hstack([b, 1])
			else:
				trajectory = lambda b0, p, t: self._trajectory(b0, p, t)
				b, info = gn(f=res, F=jac, x0=b0, 
						trajectory = trajectory, gnsolver = self._gnsolver, **self.kwargs)
				self.b = np.copy(b)

		else:
			res = lambda b: self.residual(b.view(complex), return_real=True)
			jac = lambda b: self.jacobian(b.view(complex))

			# Setup a trajectory with unit norm b
			trajectory = lambda b0, p, t: self._trajectory(b0.view(complex), p.view(complex), t).view(float)
			
			if self.normalize == 'monic':
				b0 = b0[0:-1]/b0[-1]
				bRI, info = gn(f=res, F=jac, x0=b0.view(float), **self.kwargs)
				b = bRI.view(complex)
				self.b = b = np.hstack([b, 1])
			else:	
				bRI, info = gn(f=res, F=jac, x0=b0.view(float), 
					trajectory = trajectory, gnsolver = self._gnsolver, **self.kwargs)
				b = bRI.view(complex)
				self.b = np.copy(b)

		# Compute coefficients for the numerator
		Phi = self._Phi
		Psi = self._Psi
		
		# Compute diag(Psi*b)^{-1} Phi
		Omega = (Phi.T * np.dot(Psi, b)**(-1)).T
		if self.W is None:
			WOmega = Omega
			Wh = self.h
		else:
			WOmega = np.dot(self.W, Omega)
			Wh = np.dot(self.W, self.h)

		self.a = np.linalg.lstsq(WOmega, Wh, rcond = -1)[0]
		if self.real:
			self.a = self.a.real

	def _gnsolver(self, F, f):
		U, s, VH = svd(F, full_matrices = False, compute_uv = True)
		if self.real:
			p = np.dot(U[:,:-1].T, -f)
			p = p/s[:-1]
			p = np.dot(VH[:-1,:].T,p)
			return p, s[0:-1]
		else:	
			p = np.dot(U[:,:-2].T, -f)
			p = p/s[:-2]
			p = np.dot(VH[:-2,:].T,p)
			return p, s[0:-2]

	def _trajectory(self, b, p, t):
		# Orthogonalize against scaling
		#b1 = np.copy(b)
		#b1[1::2] = 0
		#b2 = np.copy(b)
		#b2[0::2] = 0
		#Q, R = np.linalg.qr(np.vstack([b1, b2]).T, mode = 'reduced')

		# Orthogonalize search direction for safety
		Q = b/np.linalg.norm(b)
		p = p - np.dot(Q, np.dot(Q.T, p))
		# Move along geodesic, but since 1-D we have simplications
		# p has only one singular value -- its norm
		s = np.linalg.norm(p)
		# Y is simply a normalized version of p
		y = p/s
		# Z is one because of dimensions and normalization
		b_new = b*np.cos(s*t) + y*np.sin(s*t)
		return b_new/np.linalg.norm(b_new) 

	def _init_aaa(self):
		verbose = False
		try: verbose = self.kwargs['verbose']
		except KeyError: pass

		aaa = AAARationalFit(self.n, verbose = verbose)
		aaa.fit(self.z, self.h)
		lam = aaa.poles()		
		return lam		


	def numerator_vandmat(self, z = None):
		if self.numerator_basis == 'legendre':
			Phi = self.legendre_vandmat(self.m, z)
		elif self.numerator_basis == 'lagrange':
			Phi = self.lagrange_vandmat(self.zhat_numerator, z)
		return Phi

	def denominator_vandmat(self, z = None):
		if self.denominator_basis == 'legendre':
			Psi = self.legendre_vandmat(self.n, z)
		elif self.denominator_basis == 'lagrange':
			Psi = self.lagrange_vandmat(self.zhat_denominator, z)
		
		return Psi

	def _call(self, zeval):
		Phi = self.numerator_vandmat(zeval)
		Psi = self.denominator_vandmat(zeval)

		return np.dot(Phi, self.a)/np.dot(Psi, self.b)		


	def plain_residual_jacobian(self, x, return_real = False, jacobian = True):
		assert self.real is False, "plain Jacobian only implemented for complex systems"
		
		
		a = x[:self.m+1]
		b = x[self.m+1:]
		if self.normalize == 'monic':
			b = np.hstack([b,1])
			n_param = self.n
		else:
			n_param = self.n + 1

		Phi = self._Phi
		Psi = self._Psi
		
		# Compute diag(Psi*b)^{-1} Phi
		Omega = (Phi.T * np.dot(Psi, b)**(-1)).T

		mismatch = self.h - np.dot(Omega, a)
		if self.W is None:
			r = mismatch
		else:
			r = np.dot(self.W, mismatch)
		
		if not np.all(np.isfinite(r)):
			print("residual not finite!!")
			raise Exception

		if jacobian is False:
			if return_real: return r.view(float)
			else: return r
		
		JRI = np.empty( (self.z.shape[0] * 2, (self.m + 1 + n_param) * 2 ), dtype = np.float64)
		
		# Now compute the Jacobian
		La = np.array([-Phi[:,k]/np.dot(Psi, b) for k in range(self.m+1)]).T
		Lb = np.array([(np.dot(Phi,a)*Psi[:,k]/np.dot(Psi,b)**2) for k in range(n_param)]).T
		L = np.hstack([La, Lb])
		
		JRI[0::2, 0::2] = L.real 
		JRI[0::2, 1::2] = -L.imag
		JRI[1::2, 0::2] = L.imag 
		JRI[1::2, 1::2] = L.real 
		
		return r.view(float), JRI	

	def plain_residual(self, x, return_real = False):
		return self.plain_residual_jacobian(x, return_real = return_real, jacobian = False)

	def plain_jacobian(self, x):
		r, J = self.plain_residual_jacobian(x, return_real = True, jacobian = True)
		return J

	def residual_jacobian(self, b, return_real = False, jacobian = True):
		if self.normalize == 'monic':
			b = np.hstack([b, 1])
			n_param = self.n
		else:
			n_param = self.n + 1
		
		Phi = self._Phi
		Psi = self._Psi

		# Compute diag(Psi*b)^{-1} Phi
		Omega = (Phi.T * np.dot(Psi, b)**(-1)).T

		# Apply mass matrix
		if self.W is None:
			WOmega = Omega
			Wh = self.h
		else:
			WOmega = np.dot(self.W, Omega)
			Wh = np.dot(self.W, h)	

		if self.real:
			WOmegaRI = np.zeros((WOmega.shape[0]*2, WOmega.shape[1]), dtype = np.float)
			WOmegaRI[0::2,:] = WOmega.real
			WOmegaRI[1::2,:] = WOmega.imag
			WhRI = Wh.view(float)
			WOmega = WOmegaRI
			Wh = WhRI
	
		# Compute the short-form QR factorization of WV to solve system
		WOmega_Q, WOmega_R = np.linalg.qr(WOmega, mode='reduced')
		c = np.dot(WOmega_Q.conjugate().T, Wh)
		
		# First we compute the coefficients for the numerator polynomial
		a = solve_triangular(WOmega_R, c) 

		# Compute the residual		
		r = Wh - np.dot(WOmega, a) 
	
		if not np.all(np.isfinite(r)):
			print("residual not finite!!")
			raise Exception

		# Stop if we don't need to compute the jacobian
		if jacobian is False:
			if return_real and self.real: return r
			elif return_real and not self.real: return r.view(float)
			elif not return_real and self.real: return r.view(complex)
			elif not return_real and not self.real: return r

		# Now compute the Jacobian
		if self.real:
			JRI = np.empty( (self.z.shape[0] * 2, n_param), dtype = np.float64)
		else:
			JRI = np.empty( (self.z.shape[0] * 2, n_param * 2 ), dtype = np.float64)
		
		for k in range(n_param):
			# Columnwise derivative 
			dOmega = -(Phi.T * ( Psi[:,k]*np.dot(Psi, b)**(-2))).T
			if self.W is None:
				dWOmega = dOmega
			else:
				dWOmega = np.dot(W, dOmega)

			if self.real:
				dWOmegaRI = np.zeros((dWOmega.shape[0]*2, dWOmega.shape[1]), dtype = np.float)
				dWOmegaRI[0::2,:] = dWOmega.real
				dWOmegaRI[1::2,:] = dWOmega.imag
				dWOmega = dWOmegaRI

			# Compute the first term in the VARPRO Jacobian
			dWOmega_a = np.dot(dWOmega, a)
			L = -(dWOmega_a - np.dot(WOmega_Q, np.dot(WOmega_Q.conj().T, dWOmega_a)))
	
			if self.real:
				JRI[:, k] = L
			else:
				JRI[0::2, 2 * k]     = L.real
				JRI[0::2, 2 * k + 1] = -L.imag
				JRI[1::2, 2 * k]     = L.imag
				JRI[1::2, 2 * k + 1] = L.real

			# Compute the second term in the VARPRO Jacobian
			dWOmega_r = np.dot(dWOmega.conj().T, r)
			K = -np.dot(WOmega_Q, solve_triangular(WOmega_R, dWOmega_r, trans = 'C'))

			if self.real:
				JRI[:, k] += K
			else:
				JRI[0::2, 2 * k]     += K.real
				JRI[0::2, 2 * k + 1] += K.imag
				JRI[1::2, 2 * k]     += K.imag
				JRI[1::2, 2 * k + 1] += -K.real
		
		return r.view(float), JRI	

	def residual(self, b, return_real = False):
		return self.residual_jacobian(b, return_real = return_real, jacobian = False)

	def jacobian(self, b):
		r, J = self.residual_jacobian(b, return_real = True, jacobian = True)
		return J

	def pole_residue(self):
		# Compute poles
		if self.denominator_basis == 'legendre':
			lam = self.legendre_roots(self.b)
		elif self.denominator_basis == 'lagrange':
			lam = self.lagrange_roots(self.zhat_denominator, self.b)

		I = np.argsort(lam.imag)
		lam = lam[I]

		# Compute residues
		dz = 1e-5*np.exp(2j*np.pi*np.linspace(0,1,4, endpoint = False))
		rho = [ np.dot(self.__call__(lamj + dz), dz)/len(dz) for lamj in lam ]
		rho = np.array(rho)
		
		return lam, rho

	def poles(self):
		lam, rho = self.pole_residue()
		return lam

	def residues(self):
		lam, rho = self.pole_residue()
		return rho 
		

def test_jacobian():
	z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
	h = np.tan(64*z)
	
	m = 10
	n = 10
	pb = PolynomialBasisRationalFit(m,n, real = False, normalize = 'monic')
	pb.fit(z, h)

	residual = lambda x: pb.residual(x.view(complex), return_real = True)
	jacobian = lambda x: pb.jacobian(x.view(complex))
		
	b = np.random.randn(n+1) + 1j*np.random.randn(n+1)
	if pb.normalize == 'monic':
		b = b[0:-1]/b[-1]
	else:
		b = b/np.linalg.norm(b)

	err = check_jacobian(b.view(float), residual, jacobian)
	print("Error in Jacobian %5.5e" % (err,))
	assert err < 1e-7

def test_jacobian_real():
	#z = np.linspace(-1,1,1000) + 0.j
	#h = np.abs(z) + 0.j 
	
	z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
	h = np.tan(64*z)

	m = 10
	n = 10
	pb = PolynomialBasisRationalFit(m,n, real = False, verbose = True, maxiter = 100, normalize = 'monic')
	pb.fit(z, h)

	residual = lambda x: pb.residual(x, return_real = True)
	jacobian = lambda x: pb.jacobian(x)
		
	b = np.random.randn(n+1) 
	b = b/np.linalg.norm(b)
	if pb.normalize == 'monic':
		b = b[0:-1]/b[-1]

	err = check_jacobian(b, residual, jacobian)
	print("Error in Jacobian for real problem %5.5e" % (err,))
	
	b = pb.b
	if pb.normalize == 'monic':
		b = b[0:-1]/b[-1]
	err = check_jacobian(pb.b, residual, jacobian)
	print("Error in Jacobian for real problem %5.5e" % (err,))
	assert err < 1e-7


def test_plain_jacobian():
	z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
	h = np.tan(64*z)
	
	m = 9
	n = 10
	pb = PolynomialBasisRationalFit(m,n)
	pb.fit(z, h)

	residual = lambda x: pb.plain_residual(x.view(complex), return_real = True)
	jacobian = lambda x: pb.plain_jacobian(x.view(complex))
		
	a = np.random.randn(m+1) + 1j*np.random.randn(m+1)
	b = np.random.randn(n+1) + 1j*np.random.randn(n+1)
	
	if pb.normalize == 'monic':
		b = b[0:-1]/b[-1]
	else:
		b = b/np.linalg.norm(b)

	x = np.hstack([a,b])
	Jcond = np.linalg.cond(jacobian(x))
	
	err = check_jacobian(x.view(float), residual, jacobian)
	print("Error in Jacobian %5.5e" % (err,))
	assert err < 1e-10*Jcond
	
	err = check_gradient(x.view(float), residual, jacobian)
	print("Error in gradient %5.5e" % (err,))
	assert err < 1e-10*Jcond


if __name__ == '__main__':

	test_jacobian()
	#test_jacobian_real()
	#test_plain_jacobian()
	#assert False

	import scipy.io
	dat = scipy.io.loadmat('data/fig_local_minima_cdplayer.mat')
	z = dat['z'].flatten()
	h = dat['h'].flatten()
	
	pb = PolynomialBasisRationalFit(5,5, verbose = True, tol = 1e-10, 
		denominator_basis = 'legendre', numerator_basis = 'legendre', real = False, maxiter = 100,
		normalize = 'monic')
	pb.fit(z, h)
	
	b = pb.b
	a = pb.a

	print(a)
	print(b)
	
	if False:
		residual = lambda x: pb.residual(x, return_real = True)
		jacobian = lambda x: pb.jacobian(x)
			
		b = np.random.randn(pb.n+1)
		b = b/np.linalg.norm(b)
		b = b
		err = check_jacobian(b, residual, jacobian)
		print("Error in Jacobian %5.5e" % (err,))

#	pb.plain_residual_jacobian(a,b)
#
#	Phi = pb._Phi
#	Psi = pb._Psi
#
#	
#	Omega = (Phi.T / np.dot(Psi, b) ).T
#	Q, R = np.linalg.qr(Omega, mode = 'reduced')
#	
#	z = np.dot(Psi, b)/(np.dot(Psi, b)**2)
#	print z.shape
#	z = np.dot(np.diag(z), np.dot(Phi, a))
#	print np.linalg.norm(z - np.dot(Q, np.dot(Q.conj().T, z)))
#
#	b = np.random.randn(pb.n+1) + 1j*np.random.randn(pb.n +1)
#	JRI = pb.jacobian(b)
#	print np.linalg.norm(np.dot(JRI, b.view(float)))
#	U,s, VT = np.linalg.svd(JRI, compute_uv = True)
#	V = VT[-2:,:].T
#	bRI = b.view(float)
#	print np.linalg.norm(bRI - np.dot(V, np.dot(V.T, bRI)))
	#print np.dot(VT[:,-2:].T, b.view(float))
	#K = np.zeros( (z.shape[0], pb.n+1), dtype = np.complex)
	#for k in range(pb.n+1):
	#	K += 
 
	#b = np.random.randn(*pb.b.shape) + 1j*np.random.randn(*pb.b.shape)
	#b = pb.b
	#print "residual", np.linalg.norm(pb(z) - h)
	#lam, rho = pb.pole_residue()
	#print "poles", lam
	#print "residues", rho
	#JRI = pb.jacobian(b)
	#print "inner product with b", np.linalg.norm(np.dot(JRI, b.view(float)) )/np.linalg.norm(JRI)
	#print "jacobian condition", np.linalg.svd(JRI, compute_uv = False)
