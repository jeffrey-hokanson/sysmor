import numpy as np
from .ratfit import RationalFit
from .aaa import AAARationalFit
from .marriage import hungarian_sort

class BadStep(Exception):
    pass

class OptimizationRationalFit(RationalFit):
	""" Parent class for optimization based approaches for rational fitting

	This class defines two initialization heuristics
	"""
	def __init__(self, m, n, field = 'complex', stable = False, init = 'aaa',  **kwargs):
		RationalFit.__init__(self, m, n, field = field, stable = stable)

		assert init in ['recursive', 'aaa'], "Did not recognize initialization strategy"
		if init == 'recursive': self._init = self._init_recursive
		elif init == 'aaa':	self._init = self._init_aaa
		
		self.kwargs = kwargs

	def _init_aaa(self, W):
		""" Use the AAA Algorithm to initialize the poles
		"""
		aaa = AAARationalFit(self.n)
		# Should we transform data for stability?
		aaa.fit(self.z, self.f)
		lam, res = aaa.pole_residue()		

		if self.field == 'real':
			I = hungarian_sort(lam, lam.conjugate())
			lam = 0.5*(lam + lam[I].conjugate())
		# If we want stable, flip poles into LHP	
		if self.stable:
			lam = -np.abs(lam.real) + 1j*lam.imag
	
		return lam
	

	def _init_recursive(self, W):
		m_orig = self.m
		n_orig = self.n
		if self.real: 
			step = 2
			if n_orig % 2 == 0: 
				# Even number of poles
				n = 2
			else:	
				# Odd number of poles
				n = 1
		else:
			n = 1 
			step = 1

		m = (m_orig - n_orig) + n 
		#m = n - 1
		
		# Constants we will need later
		max_real = np.max(self.z.real)
		min_real = np.min(self.z.real)
		max_imag = np.max(self.z.imag)
		min_imag = np.min(self.z.imag)
		
		# For the denominator we will use a barycentric polynomial
		# with nodes at the corners of the real or imaginary range 
		# (whichever is largest)
		if (max_real - min_real) > (max_imag - min_imag):	
			nodes = [min_real, max_real]
			nodes += [ (max_real + min_real)/2]
			nodes = np.array(nodes) + 1j*(min_imag + max_imag)/2.
		else:
			nodes = [1j*min_imag, 1j*max_imag]
			nodes += [ 1j*(min_imag + max_imag)/2. ]
			nodes = np.array(nodes) + (min_real + max_real)/2.
	
		lam = None
		res = np.copy(self.f)
		while n <= n_orig:
			num_degree = min(step, n)+1
			q = LagrangePolynomial(nodes[:num_degree])
			Psi = q.vandmat(self.z)

			p = LagrangePolynomial(nodes[:num_degree])
			Phi = p.vandmat(self.z)

			# Setup the "linearized" solution
			A = np.hstack([Phi, -np.dot(np.diag(res), Psi) ])
			A = self.W(A)
		
			#print "cond A %5.5e, cond Phi %5.5e, cond Psi %5.5e" % ( 
			#	np.linalg.cond(A), np.linalg.cond(Phi), np.linalg.cond(Psi))
			U, s, VH = np.linalg.svd(A)
			ab = VH.conjugate().T[:,-1]
			
			# Compute the new roots to add
			b = ab[-num_degree:]
			q = LagrangePolynomial(nodes[:num_degree], b)
			lam_new = q.roots()
			#lam_new = self.legendre_roots(b)
			#print lam_new
			# Force the new roots to come in conjugate pairs
			if self.real:
				I = hungarian_sort(lam_new, lam_new.conjugate())
				lam_new = 0.5*(lam_new + lam_new[I].conjugate()) 

			# Either use these roots as the initial values 
			if lam is None: lam = lam_new
			# Or append them to the current list of roots
			else: lam = np.hstack([self.lam, lam_new])
			
			#I = np.argsort(lam.imag)
			#print "poles", lam[I]
			#print "lam new", lam_new
			
			# Fit to the current size
			self.m = m
			self.n = n
			if self.n == n_orig:
				#print "stopping with (%d,%d)" % (self.m, self.n)
				break

			self._fit(lam)
			res = self.residual(self.lam)
			#print "fitting (%d, %d)" % (m,n), "residual", np.linalg.norm(res)
			m += step
			n += step

		self.m = m_orig
		self.n = n_orig
		return lam				
