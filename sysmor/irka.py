from __future__ import division, print_function
import numpy as np

# For delay systems, see: https://people.kth.se/~eliasj/doc/delaylyap-2010-07-12.pdf
from scipy.linalg import eig, expm, block_diag, eigvals
from scipy.sparse import issparse
from scipy.sparse import eye as speye
from scipy.sparse.linalg import eigs, spsolve

from .h2mor import H2MOR
from .system import StateSpaceSystem, PoleResidueSystem, ZeroSystem
from .marriage import hungarian_sort, marriage_norm
	

def rational_krylov_approximation(H, mu):
	r""" Constructs a rational Krylov approximation

	By construction this recovers a real system.


	Parameters
	----------
	H: StateSpaceSystem
		State space FOM to reduce
	mu: array-like
		Shift locations

	Returns
	-------
	Hr: StateSpaceSystem	
		Reduced order model interpolating at mu
	nsolve: int
		Number of linear solves
	"""
	assert isinstance(H, StateSpaceSystem), "IRKA only applies to state-space systems"

	n = H.state_dim
	r = mu.shape[0]

	V = np.zeros((n, r))
	W = np.zeros((n, r))

	assert r % 2 == 0, "Degree r must be even, r = %d" % r

	B = H.B
	C = H.C
	i = 0
	nsolves = 0
	while i < r:
		if np.abs(mu[i].imag) / np.abs(mu[i]) < 1e-10:
			v = H.solve(B, mu[i].real)
			w = H.solve(C.T, mu[i].real, mode = 'T')
			nsolves += 2
			V[:, i] = v.flatten()
			W[:, i] = w.flatten()
			i += 1
		else:
			v = H.solve(B, mu[i])
			w = H.solve(C.T, mu[i], mode = 'T')
			nsolves += 2
			V[:, i:i+2] = np.c_[v.real, v.imag]
			W[:, i:i+2] = np.c_[w.real, w.imag]
			i += 2

	V = np.linalg.svd(V, full_matrices = 0)[0]
	W = np.linalg.svd(W, full_matrices = 0)[0]

	Er = W.T.dot(H.E.dot(V))
	W = np.linalg.solve(Er, W.T).T

	# Construct ROM
	Ar = W.T.dot(H.A.dot(V))
	Br = W.T.dot(H.B)
	Cr = H.C.dot(V)
	Hr = StateSpaceSystem(Ar, Br, Cr)
	return Hr, nsolves 

class IRKA(H2MOR, StateSpaceSystem):
	"""
	Parameters
	----------
	xtol: float, optional
		stopping tolerance
	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000, flipping = True, verbose = True, ftol = 1e-7, lamtol = 1e-6, print_norm = True):
		H2MOR.__init__(self, rom_dim, real = real)
		assert self.real, "Implementation only handles real approximating systems"
		#assert rom_dim % 2 == 0, "Only even recovered systems currently supported"
		self.maxiter = int(maxiter)
		self.flipping = flipping
		self.verbose = verbose
		self.ftol = ftol
		self.lamtol = lamtol
		self.print_norm = print_norm

	def _mu_init(self, H):
		# Copmute the poles closest to the real line as an initialization
		lam = H.poles(which = 'LR', k = self.rom_dim)
		mu0 = -lam.conj()
		
		if self.real:
			I = hungarian_sort(mu0, mu0.conj())
			mu0 = 0.5*(mu0 + mu0[I].conj())

		return mu0


	def _fit_iterate(self, H, mu):
		# Compute new rational interpolant based on shifts mu
		assert H.isreal
		Hr, nsolves = rational_krylov_approximation(H, mu)
		self._total_linear_solves += nsolves
		return Hr

	def _fit(self, H, mu0 = None):
		if mu0 is None:
			mu0 = self._mu_init(H)
		else:
			assert len(mu0) == self.rom_dim, "Must provide as many initial shifts as the ROM dimension"

		mu = np.copy(mu0)
		Hr = ZeroSystem(H.output_dim, H.input_dim)

		for it in range(0, self.maxiter):
			Hr_old = Hr
			mu_old = np.copy(mu)

			# Construct an interpolant given the current estimate of the Meier-Luenberger interpolation points
			Hr = self._fit_iterate(H, mu) 

			# Flip poles into LHP and reconstruct system
			used_flip = False
			if self.flipping:
				lam, rho = Hr.pole_residue()
				# If their are poles in the RHP, flip back into LHP, keeping residues
				if np.max(lam.real) >= 0:
					I = lam.real > 0
					lam[I] = -lam[I].conj()
					# TODO: Is this right way to handle the residues?
					rho[I] = rho[I].conj()
					Hr = PoleResidueSystem(lam, rho)
					used_flip = True


			if self.history is not None:
				self.history.append({
					'mu': np.copy(mu), 
					'Hr': Hr, 
					'total_fom_evals': self._total_fom_evals,
					'total_fom_der_evals': self._total_fom_der_evals,
					'total_linear_solves': self._total_linear_solves,
				})

			# Compute new shifts mu
			lam = Hr.poles()
			# Flip into RHP
			mu = np.abs(lam.real) + 1j* lam.imag 
		
			# If seeking a real ROM, force poles/residues to come in conjugate pairs 
			if self.real:
				I = hungarian_sort(mu, mu.conj())
				mu = 0.5*(mu + mu[I].conj())
				# Ensure they are accurate to all bits
				mu[mu.imag < 0 ] = mu[mu.imag > 0].conj()

			I = hungarian_sort(mu, mu_old)
			delta_mu = np.linalg.norm(mu - mu_old[I])	
		
			Hr_norm = Hr.norm()
			res_norm = (Hr - Hr_old).norm()	

			if self.verbose:
				self._iter_message(it, res_norm, Hr_norm, delta_mu, used_flip, H, Hr)

			# Check stopping conditions
			if res_norm/Hr_norm < self.ftol/Hr_norm:
				if self.verbose:
					print("Stopped due to small movement of Hr")
				break
			
			if delta_mu < self.lamtol:
				if self.verbose:
					print("Stopped due to small movement of poles")
				break
	
		# Copy over ROM to this instance
		StateSpaceSystem.__init__(self, Hr.A, Hr.B, Hr.C)

	def _iter_message(self, it, res_norm, Hr_norm, delta_mu, used_flip, H, Hr):
		if it == 0:
			head1 = "  it | Lin Solves | delta Hr | delta mu | flipped |"
			head2 = "-----|------------|----------|----------|---------|"
			if self.print_norm:
				head1 += ' Error H2 Norm |'
				head2 += '---------------|'
			print(head1)
			print(head2)

		iter_message = "%4d |    %7d | %8.2e | %8.2e | %7s |" % \
			(it, self._total_linear_solves, res_norm/Hr_norm, delta_mu, used_flip) 
		
		if self.print_norm:
			iter_message += ' %13.6e |' % ( (H-Hr).norm()/H.norm())

		print(iter_message)
	

if __name__ == '__main__':
	from demos import build_iss
	H = build_iss()
	H = H[0,0]
	Hr = IRKA(rom_dim = 50, maxiter = 100, ftol = 1e-9)
	Hr.fit(H)
	print (H - Hr).norm()/H.norm()

