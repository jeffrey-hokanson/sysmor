import numpy as np
import scipy.linalg
from .system import PoleResidueSystem, ZeroSystem
from .h2mor import H2MOR

class QuadVF(H2MOR, PoleResidueSystem):
	"""Quadrature based H2 Model Reduction
	
	

	Sources:
	[DGB15]: Quadrature-based Vector Fitting for Discretized H2 Approximation, Z. Drmac, S. Gugercin, and C. Beattie,
		SIAM J. Sci. Comput., Vol. 37, No. 2 p.A625--A652, 2015

	Parameters
	----------
	rom_dim: int
		Dimension of ROM to build
	N: int
		Number of samples to take; if real
	L: float, positive
		Parameter controlling the scaling of the quadrature rule
	real: bool
		If true, construct a real reduced order model
	"""
	def __init__(self, rom_dim, N, L = 1, real = True, maxiter = 100, verbose = False, btol = 1e-10, ftol = 1e-10):
		
		self.rom_dim = rom_dim
		self.L = np.abs(L)
		self.N = int(N)
		self.maxiter = maxiter
		self.btol = btol
		self.ftol = ftol
		
		self.real = real
		self.verbose = verbose
		self.history = []

	def _fit(self, H, lam0 = None, mu0 = None):
		if H.isreal:
			# As H is real, we can half the number of samples used in the Vector Fitting step
			return self._fit_real(H, lam0 = lam0)
		else:
			raise NotImplementedError

	def _fit_real(self, H, lam0 = None):
		if H.input_dim > 1 or H.output_dim > 1:
			raise NotImplementedError

		# Construct the locations to sample with positive real part
		mu = self.L*1.j/np.tan(np.arange(1,2*self.N+1)*np.pi/(2*self.N+1))
		# Ensure these are numerically conjugate pairs to all digits
		mu = 0.5*(mu + mu[::-1].conj())
	
	
		if lam0 is None:
			# Estimate where to place starting poles
			# Alternatively, should we use log spaced like in DGB15, p. A632?
			lam0 = -1e-5+ 1j*np.linspace(np.min(mu.imag), np.max(mu.imag), self.rom_dim)

		# Sample, invoking conjugacy
		h = self.eval_transfer(H, mu).flatten()
		
		# Add the 'sample' at infinity
		h = np.hstack([h, np.asscalar(H.lim_zH[1].flatten())])

		# Compute weights of quadrature rule
		# Equivalent of the mass matrix; see eq. (3.7) DGB15	
		Delta = 1./np.sin(np.arange(1, 2*self.N+1)*np.pi/(2*self.N + 1))*np.sqrt(self.L*np.pi/(2*self.N+1))
		# Add the special weight at the bottom
		Delta = np.hstack([Delta, np.sqrt(np.pi/(self.L*(2*self.N+1)))])

		# Apply vector fitting
		lam = np.copy(lam0)
		n = len(mu)
		r = len(lam)
		Hr = ZeroSystem(1,1)
		# TODO: Ideally, we should modify the vector fitting code to allow this weighting
		# and additional rows.  
		# TODO: Although not what is done in DGB15, we should ideally 
		# enforce the requirement for real systems using the technique present in the original vector fitting paper
		for it in range(self.maxiter):
			Hr_old = Hr

			# Build A matrix eq. (3.7) DGB15		
			A1 = 1./(np.tile(mu.reshape(n,1), (1,r)) - np.tile(lam.reshape(1,r), (n,1)))
			A2 = -h[:-1].reshape(-1,1)*1./(np.tile(mu.reshape(n,1), (1,r)) - np.tile(lam.reshape(1,r), (n,1)))
			A = np.hstack( [A1, A2])

			# Add the additional row for the point at infinity
			A = np.vstack( [ A, np.hstack( [np.ones(self.rom_dim), np.zeros(self.rom_dim)]).reshape(1,-1) ])

			# Apply the diagonal weighting 
			DA = np.dot(np.diag(Delta), A)
			Dh = np.dot(np.diag(Delta), h)

			# Solve the linear system			
			x, res, rank, A_sing = scipy.linalg.lstsq(DA, Dh, lapack_driver = 'gelss')
			a = x[:self.rom_dim]
			b = x[self.rom_dim:]

			Hr = PoleResidueSystem(lam, a)	
			Hr_norm = Hr.norm()
			delta_Hr = (Hr - Hr_old).norm()/Hr_norm
			
			if self.verbose:
				if it == 0:
					head1 = "  it |  A cond  |   norm b | delta Hr |"
					head2 = "-----|----------|----------|----------|"
					print(head1)
					print(head2)
				iter_message = "%4d | %8.2e | %8.2e | %8.2e |" % (it, A_sing[0]/A_sing[-1], np.linalg.norm(b), delta_Hr)
				print(iter_message)

			if np.linalg.norm(b) < self.btol:
				if self.verbose:
					print("Stopped due to small change in b")
				break
			
			if delta_Hr < self.ftol/Hr_norm:
				if self.verbose:
					print("Stopped due to small change in Hr")
				break 	
			
			# Update the poles 
			if it < self.maxiter -1:
				lam = np.linalg.eigvals(np.diag(lam) - np.outer(np.ones(lam.shape), b))

			
		# Copy over data
		rho = a	
		if np.max(lam.real) >= 0:
			I = lam.real > 0
			lam[I] = -lam[I].conj()
			# TODO: Is this right way to handle the residues?
			rho[I] = rho[I].conj()

		
		self.history.append({
			'mu': np.copy(mu), 
			'Hr': PoleResidueSystem(lam, rho), 
			'total_fom_evals': self._total_fom_evals,
			'total_fom_der_evals': self._total_fom_der_evals,
			'total_linear_solves': self._total_linear_solves,
		})

		PoleResidueSystem.__init__(self, lam, rho)

if __name__ == '__main__':
	from demos import build_iss
	H = build_iss()
	H = H[0,0]
	Hr = QuadVF(rom_dim = 10, N = 1e2, L = 10, maxiter = 100)
	Hr.fit(H)
	print (H - Hr).norm()/H.norm()

