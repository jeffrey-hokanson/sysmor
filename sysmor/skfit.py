import numpy as np
import scipy
from .ratfit import RationalFit
from .pbfit import PolynomialBasisRationalFit
from scipy.linalg import lstsq

class SKRationalFit(PolynomialBasisRationalFit):
	def __init__(self, m, n, numerator_basis = 'legendre', denominator_basis = 'legendre',
		W = None, zhat_numerator = None, zhat_denominator = None, maxiter = 100, verbose = False, tol = 1e-10,
		init = 'linearize', svd = True, normalize = 'monic'):

		self.m = m
		self.n = n

		self.real = False
		self.W = W

		self.numerator_basis = numerator_basis
		self.denominator_basis = denominator_basis

		self.zhat_numerator = zhat_numerator
		self.zhat_denominator = zhat_denominator

		assert init in ['linearize', 'aaa']
		self.init = init
		
		assert normalize in ['svd', 'monic']
		self.normalize = normalize

		self.maxiter = maxiter
		self.verbose = verbose
		self.tol = tol
		self.kwargs = {}
		self.svd = svd

	def _fit(self, lam0):
		# Setup Lagrange nodes if these have not been specified
		if self.numerator_basis == 'lagrange' and self.zhat_numerator is None:
			self.zhat_numerator = self._generate_zhat(self.m)
		if self.denominator_basis == 'lagrange' and self.zhat_denominator is None:
			self.zhat_denominator = self._generate_zhat(self.n)

		self._Phi = self.numerator_vandmat()
		self._Psi = self.denominator_vandmat()


		A = np.hstack([self._Phi, -(self._Psi.T*self.h).T])

		def compute_ab(A):
			if self.normalize == 'svd':
				U, s, VH = scipy.linalg.svd(A, full_matrices = False, compute_uv = True)
				x = VH.conj().T[:,-1]
			elif self.normalize == 'monic':
				# Check me!
				I = np.ones(A.shape[1], dtype = np.bool)
				I[self.m+1] = 0
				x = np.ones(self.m+self.n+2, dtype = np.complex)
				x[I] = lstsq(A[:,I], -A[:,~I])[0].flatten()
			else:
				raise NotImplementedError
			a = x[:self.m+1]
			b = x[self.m+1:]
			return a, b	
	
		if self.init == 'linearize':
			self.a, self.b = compute_ab(A)	
		elif self.init == 'aaa':
			if self.verbose:
				print("Initializing SK iteration with AAA")
			lam0 = self._init_aaa()
			self.b = self._convert_lam0(lam0)


		for it in range(self.maxiter):
			Ahat = (A.T/np.dot(self._Psi,self.b)).T
			a, b = compute_ab(Ahat)	
			
			# Since we have a free rotation and scaling,
			# we fix ||b||=1 and require first term to have real sign
			# b_norm = np.linalg.norm(b)*(b[0]/np.abs(b[0]))
			scale = 1./np.linalg.norm(b)
			b *= scale
			a *= scale
	
			# We compute error by projection norm (removing the need to compenstate for scaling)
			move = np.linalg.norm(self.b - np.dot(b, np.dot(b.conj().T, self.b)))
		
			#move = np.linalg.norm(b - self.b, np.inf)
			#angle = np.abs(np.dot(b.conj().T, self.b)/(np.linalg.norm(b)*np.linalg.norm(self.b)))
			#angle = 180./np.pi*np.arccos(min(max(angle,0),1))
			if self.verbose:
				x = np.hstack([a,b])
				r, J = self.plain_residual_jacobian(x)
				g = np.dot(J.T, r)
				print("%3d delta-b %5.5e, ||r|| %5.5e gradient norm %5.5e" % (it, move, np.linalg.norm(r), np.linalg.norm(g)))
			
			# Update values
			self.a = a	
			self.b = b
			if move < self.tol:
				break
		


if __name__ == '__main__':
	import scipy.io
	dat = scipy.io.loadmat('data/fig_aaa_cdplayer.mat')
	z = dat['z'].flatten()
	h = dat['h'].flatten()

	#z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
	#h = np.tan(64*z)

	z = z[::10]
	h = h[::10]

	m = 10
	n = 11	

	sk = SKRationalFit(m,n, verbose = True, tol = 1e-10, 
		denominator_basis = 'legendre', numerator_basis = 'legendre', init = 'linearize', normalize = 'monic')
	
	pb = PolynomialBasisRationalFit(m,n, verbose = True, tol = 1e-12, maxiter = 50, 
		denominator_basis = 'legendre', numerator_basis = 'legendre')

	sk.fit(z,h)
	print("residual norm (SK): %5.5e" % np.linalg.norm(sk(z) - h))
	# Check gradient at optimizer:
	# Albeit this is the VARPRO-ed Jacobian
	x = np.hstack([sk.a, sk.b])
	r, J = sk.plain_residual_jacobian(x)
	print("gradient norm (SK): %5.5e" % (np.linalg.norm(np.dot(J.T, r)),))

	# Same for pb
	pb.fit(z,h)
	r, J = pb.residual_jacobian(pb.b)
	print("residual norm (PB): %5.5e" % np.linalg.norm(pb(z) - h))
	print("gradient norm (PB): %5.5e" % (np.linalg.norm(np.dot(J.T, r)),))

	print("mismatch (PB) - (SK): %5.5e" % (np.linalg.norm(pb(z) - sk(z)),))	
