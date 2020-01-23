""" Numerical example from Model Reduction and Approximation, Ex. 7.14, p 322
"""
import numpy as np
import scipy.linalg 
import mor, mor.demos
from mor.pgf import PGF
import argparse
from scipy.signal import find_peaks, convolve


import matplotlib.pyplot as plt
import scipy.sparse.linalg



def run_mor(MOR, rs, prefix, **kwargs):
	N_norm = 1e3		# Number of samples for quadrature norms

	H = mor.demos.build_bg_delay(chi = 10, tau = 0.1, n = 100)
	
	H_norm = H.quad_norm(n = N_norm, L = 10)

	rel_err = np.nan*np.zeros(rs.shape)	
	fom_evals = np.nan*np.zeros(rs.shape)
		
	## Compute the initial shifts
	#A0 = -(H.A0 + H.A1).toarray()
	#A1 = (H.tau*H.A1 + H.E).toarray()
	#A2 = (-H.tau**2/2*H.A1).toarray()
	#Z = np.zeros(A0.shape)
	#I = np.eye(A0.shape[0])
	
	#A = np.vstack([np.hstack([A1,A0]), np.hstack([-I, Z])])
	#B = np.vstack([np.hstack([A2,Z]), np.hstack([Z, I])])

	#ew = scipy.linalg.eigvals(-A, B)
	#ew = ew[np.argsort(-ew.real)]
	#ew = ew[(ew.imag != 0) & (ew.real< 0)]
	#print(ew)

	
	z = 1j*np.logspace(-1, 3, 600)
	Hz = H.transfer(z)
	
	peaks, prop = find_peaks(np.log10(np.abs(Hz).flatten()), rel_height = 1e-2)
	print(peaks, z[peaks])

	if True:
		fig, ax = plt.subplots(1,1)
		mean = convolve(np.log10(np.abs(Hz).flatten()), np.ones(100)/100, mode = 'same')
		print(np.abs(Hz).flatten() - mean) 
		ax.plot(z.imag, np.abs(Hz).flatten() - 10**mean)
		for p in peaks:
			ax.plot(z[p].imag, np.abs(Hz).flatten()[p], 'r.')
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.grid(True)
		plt.show()



	for i, r in enumerate(rs):
		np.random.seed(0)
		
	
		Hr = MOR(r, **kwargs)
		mu0 = -np.abs(ew[0:r].real) + 1j*ew[0:r].imag
		Hr.fit(H, mu0 = mu0)
		err = (H - Hr).norm()/H_norm
		
		print("%3d :err %15.10e" % (r, err))
		
		# Save information about overall performance
		rel_err[i] = err
		fom_evals[i] = Hr.history[-1]['total_fom_evals'] 
		fom_evals[i] += Hr.history[-1]['total_fom_der_evals'] 
		fom_evals[i] += Hr.history[-1]['total_linear_solves'] 

		pgf = PGF()
		pgf.add('rom_dim',rs)
		pgf.add('fom_evals', fom_evals)
		pgf.add('rel_err', rel_err)
		pgf.write(prefix+'.dat')




if __name__ == '__main__':

	# Bode plots


	rs = np.array([2])
	ftol = 1e-9	
	MOR = mor.ProjectedH2MOR
	run_mor(MOR, rs, 'data/fig_delay_ph2', verbose = 10, ftol = ftol)
	


