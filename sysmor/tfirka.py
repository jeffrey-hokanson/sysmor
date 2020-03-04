from __future__ import division
import numpy as np
import scipy.linalg
import warnings
from itertools import product

from .system import StateSpaceSystem
from .irka import IRKA

def hermite_interpolant(z, Hz, Hpz):
	r""" Construct a Hermite interpolant system given Hermite data
	"""
	# TODO: Implement MIMO Hermite Interpolants
	assert len(Hz.shape) == 1 and len(Hpz.shape) == 1, "Require scalar data"

	r = len(z)
	Ar = np.zeros((r, r), dtype=complex)
	Er = np.zeros((r, r), dtype=complex)
	Br = np.zeros((r, 1), dtype=complex)
	Cr = np.zeros((1, r), dtype=complex)

	for i, j in product(range(0, r), range(0, r)):
		if i == j:
			Ar[i,j] = - Hz[i] - z[i] * Hpz[i]
			Er[i,j] = - Hpz[i]
		else:
			Ar[i,j] = - (z[i] * Hz[i] - z[j] * Hz[j]) / (z[i] - z[j])
			Er[i,j] = - (Hz[i] - Hz[j]) / (z[i] - z[j])
	Br = Hz
	Cr = Hz

	with warnings.catch_warnings():
		warnings.simplefilter('ignore', scipy.linalg.LinAlgWarning)
		# Now invert E to make into a state-space system
		# This can be ill-conditioned early on, but generally resolves itself
		Ar = scipy.linalg.solve(Er, Ar)
		Br = scipy.linalg.solve(Er, Br)
	Hr = StateSpaceSystem(Ar, Br, Cr)
	Hr.mu = np.copy(z)
	return Hr


class TFIRKA(IRKA):
	"""
	Parameters
	----------
	xtol: float, optional
		stopping tolerance
	"""
	def __init__(self, rom_dim, real = True, maxiter = 1000, flipping = True, verbose = True, ftol = 1e-7, lamtol = 1e-6, print_norm = False):
		IRKA.__init__(self, rom_dim, real = real, maxiter = maxiter,
			 flipping = flipping, verbose = verbose, ftol = ftol, lamtol = lamtol, print_norm = print_norm)

	def _fit_iterate(self, H, mu):
		H_mu, Hp_mu = self.eval_transfer(H, mu, der = True)
		return hermite_interpolant(mu, H_mu.flatten(), Hp_mu.flatten())

	def _iter_message(self, it, res_norm, Hr_norm, delta_mu, used_flip, H, Hr):
		if it == 0:
			head1 = "  it | H(z) evals | H'(z) evals | delta Hr | delta mu | flipped |"
			head2 = "-----|------------|-------------|----------|----------|---------|"
			if self.print_norm:
				head1 += ' Error H2 Norm |'
				head2 += '---------------|'
			print(head1)
			print(head2)

		iter_message = "%4d |    %7d |     %7d | %8.2e | %8.2e | %7s |" % \
			(it, self._total_fom_evals, self._total_fom_der_evals, res_norm/Hr_norm, delta_mu, used_flip) 
		
		if self.print_norm:
			iter_message += ' %13.6e |' % ( (H-Hr).norm()/H.norm())

		print(iter_message)
 


if __name__ == '__main__':
	from demos import build_iss
	H = build_iss()
	H = H[0,0]
	
	Hr = TFIRKA(50, maxiter=100)
	Hr.fit(H)
	print (H - Hr).norm()/H.norm()
