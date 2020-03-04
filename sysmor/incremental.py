import numpy as np
import scipy.linalg as sl
from scipy.optimize import linear_sum_assignment 

from .marriage import hungarian_sort 
from .tfirka import hermite_interpolant, TFIRKA


class IncrementalTFIRKA(TFIRKA):
	r""" A modification of TF-IRKA to use 
	"""
	def _select_shifts(self, Hr, mu_old):
		lam = Hr.poles()

		if np.any(lam.imag == 0) or np.any(mu_old.imag == 0):
			mu = -lam.conj()
			I = hungarian_sort(mu, mu.conj())
			mu = 0.5*(mu + mu[I].conj())
			# Ensure they are accurate to all bits
			mu[mu.imag < 0 ] = mu[mu.imag > 0].conj()
			return mu 
		
		angle_matrix = np.zeros((len(lam), len(lam)) ) 
	
		Mhat = np.zeros((2,2), dtype = np.complex)
		Mlam = np.zeros((2,2), dtype = np.complex)
		Mmu  = np.zeros((2,2), dtype = np.complex)
		for i, lam_i in enumerate(lam):
			# Flip mu into LHP
			for j, mu_j in enumerate(-mu_old.conj()):
			#for j, mu_j in enumerate([lam_i,]):
				Mlam[0,0] = -(lam_i.conj() + lam_i)**(-1)
				Mlam[0,1] = -(lam_i.conj() + lam_i)**(-2)
				Mlam[1,0] = Mlam[0,1].conj()
				Mlam[1,1] = -2*(lam_i.conj() + lam_i)**(-3)
				
				Mmu[0,0] = -(mu_j.conj() + mu_j)**(-1)
				Mmu[0,1] = -(mu_j.conj() + mu_j)**(-2)
				Mmu[1,0] = Mmu[0,1].conj()
				Mmu[1,1] = -2*(mu_j.conj() + mu_j)**(-3)
				
				# Inner product of the two different spaces 
				Mhat[0,0] = -(lam_i.conj() + mu_j)**(-1)
				Mhat[0,1] = -(lam_i.conj() + mu_j)**(-2)
				Mhat[1,0] = Mhat[0,1].conj()
				Mhat[1,1] = -2*(lam_i.conj() + mu_j)**(-3)

				Llam = sl.sqrtm(Mlam) #sl.cholesky(Mlam, lower = False)
				Lmu = sl.sqrtm(Mmu) #sl.cholesky(Mmu, lower = False)
				A = sl.solve(Lmu, Mhat.conj().T).conj().T
				A = sl.solve(Llam, A)
				#A = sl.solve_triangular(Lmu, Mhat.conj().T, lower = False, trans = 'C').conj().T
				#A = sl.solve_triangular(Llam, A)
				#print("A", A)
				#print(sl.svdvals(A))
				# Largest subspace angle
				angle_matrix[i,j] = sl.svdvals(A)[0]

		# Since in exact arithmatic subspace angle should not exceed 1, we clip those values
		angle_matrix = np.minimum(angle_matrix, 1)
		# We could work with the angles directly, but here we work with their cosines in angle_matrix
		#angle_matrix = np.arccos(angle_matrix)*180/np.pi

		# To find the best place to sample.
		# Here we use the linear sum assignment problem to find a path through the angle matrix
		# which maximizes the sum
		# This is important because we do not want to discard accurate information we already have
		# as we might be tempted to do greedily
		row, col = linear_sum_assignment(-angle_matrix)

		# Now in this sum, we find the worst performing entry and replace it
		k = np.argmin([angle_matrix[i,j] for i,j in zip(row, col)])

		i = row[k]
		j = col[k]	
		
		#print(angle_matrix)
		mu = np.copy(mu_old)
		mu[j] = -lam[i].real + 1j*lam[i].imag

		if not self.real:
			return mu

		# If we've replaced a real sample with a real sample, we are done
		if mu[j].imag == 0 and mu_old[j].imag == 0:
			return mu
		# If we've replaced a complex sample with a complex one, respect symmetry
		elif mu_old[j].imag != 0 and mu[j].imag !=0:
			k = np.argmin(np.abs(mu_old[j] - mu_old.conj()))
			mu[k] = mu[j].conj()
			return mu
		# If we've replaced a complex conjugate pole with a real one
		#elif mu_old[j].imag !=0 and mu[j].imag == 0:
		#	angle_matrix[lam.imag != 0,:] = 0

		print(mu)
		raise NotImplementedError


