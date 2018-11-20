from __future__ import division
import numpy as np
import scipy.linalg
import warnings
from itertools import product

from system import StateSpaceSystem
from irka import IRKA

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

	with warnings.catch_warnings(scipy.linalg.LinAlgWarning):
		# Now invert E to make into a state-space system
		# This can be ill-conditioned early on, but generally resolves itself
		Ar = scipy.linalg.solve(Er, Ar)
		Br = scipy.linalg.solve(Er, Br)
	Hr = StateSpaceSystem(Ar, Br, Cr)
	return Hr


class TFIRKA(IRKA):
	"""
	Parameters
	----------
	xtol: float, optional
		stopping tolerance
	"""
	def __init__(self, rom_dim, real = True, maxiter = 50, flipping = True, verbose = True, ftol = 1e-7, lamtol = 1e-6, print_norm = True):
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
 

def test_build_Hermite():
	#Checks whether H_r(s) is in fact a Hermite interpolant at z
	H = build_cdplayer()
	rom = TFIRKA()
	z = np.array([1+4j, 1-4j, 2+1j, 2-1j], dtype=complex)
	Ar, Br, Cr, Er = rom.build_Hermite(z, H)

	z = z[2]
	EA = z * Er - Ar
	x = np.linalg.solve(EA, Br)
	Hr_z = np.dot(Cr, x)
	xp = np.linalg.solve(EA, Er.dot(x))
	Hpr_z = np.dot(-Cr, xp)
	H_z, Hp_z = H.transfer_der(z)

	#Should be numerically 0
	print "|H (z) - Hr (z)| = %1.2e" % np.abs(Hr_z - H_z)[0, 0]
	print "|H'(z) - Hr'(z)| = %1.2e" % np.abs(Hpr_z - Hp_z)[0, 0]

if __name__ == '__main__':
	from demos import build_iss
	H = build_iss()
	H = H[0,0]
	
	r = 6
	#Bug: Numerical issues arise in the r >= 18 case
	#Maybe this isn't a bug. It could be a feature of TFIRKA...
	Hr = TFIRKA(50, maxiter=100)
	#:mu = np.hstack([1 + 1j * np.arange(1,(r+1)/2), 1 - 1j * np.arange(1,(r+1)/2)])
	Hr.fit(H)
	print (H - Hr).norm()/H.norm()
