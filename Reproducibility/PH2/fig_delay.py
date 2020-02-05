""" Numerical example from Model Reduction and Approximation, Ex. 7.14, p 322
"""
import numpy as np
import scipy.linalg 
import mor, mor.demos
from mor.pgf import PGF
import argparse
from scipy.optimize import minimize

import matplotlib.pyplot as plt

n_quad = int(1e4) 	# Number of samples for quadrature norms

def run_mor(MOR, rs, prefix, hist = False, **kwargs):

	H = mor.demos.build_bg_delay()
	
	H_norm = H.quad_norm(n = n_quad, L = 10)

	rel_err = np.nan*np.zeros(rs.shape)	
	fom_evals = np.nan*np.zeros(rs.shape)
		
	z = 1j*np.logspace(-1, 3, 600)
	#z = 1j*np.linspace(0, 200, 1000)
	Hz = H.transfer(z)


	# Find poles to initialize algorithms
	poles = []
	for i in range(max(np.max(rs)//2, 4) ):
		x0 = [-0.22*(i+1), 2.83* (2.21*i+0.75)]
		obj = lambda x: -np.log10(np.abs(H.transfer(x[0] + 1j*x[1])).flatten())
		res = minimize(obj, x0, bounds = [(None,0), ( 2.83*(2.21*i-0.25), 2.83*(2.21*i+1.75) )], method = 'COBYLA')
		#res = minimize(obj, x0, bounds = [(None,0), ( None, None)])
		print(res)
		poles.append(res.x[0] + 1j*res.x[1])

	poles = np.array(poles)
	print(poles)

	if False:
		fig, ax = plt.subplots(1,1)
		ax.plot(z.imag, np.abs(Hz).flatten())
		ax.set_yscale('log')
		#ax.set_xscale('log')
		ax.grid(True)
		for lam in poles:
			ax.axvline(lam.imag, color = 'r', alpha = 0.2)
		plt.show()
		assert False

	for i, r in enumerate(rs):
		np.random.seed(0)
		
	
		Hr = MOR(r, **kwargs)
		if isinstance(Hr, mor.ProjectedH2MOR) and r == 2:
			# This is a hack to ensure overdetermined
			r = 4
		mu0 = np.array([ np.abs(poles[0:r//2].real) + 1j*poles[0:r//2].imag])
		mu0 = np.hstack([mu0, mu0.conj()]).flatten()
	
		Hr.fit(H, mu0 = mu0)
		err = (H - Hr).quad_norm(L = 10, n = n_quad)/H_norm
		
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

		# Generate Bode plot
		Hrz = Hr.transfer(z)
		pgf = PGF()
		pgf.add('z', z.imag)
		pgf.add('Hz', np.abs(Hz).flatten())
		pgf.add('Hrz', np.abs(Hrz).flatten())
		pgf.add('diff', np.abs(Hz - Hrz).flatten())
		pgf.write(prefix + '_bode_%02d.dat' % r)

		if hist:
			print("History")
			fom_eval_hist = []
			rel_err_hist = []
			for hist in Hr.history:
				fom_eval_hist.append(hist['total_fom_evals'] + hist['total_fom_der_evals'] + hist['total_linear_solves'])
				rel_err_hist.append( (H - hist['Hr'] ).quad_norm(n = n_quad, L = 10)/H_norm )
				pgf = PGF()
				pgf.add('fom_evals', fom_eval_hist)
				pgf.add('rel_err', rel_err_hist)
				pgf.write(prefix + '_hist_%02d.dat' % r)
		

def run_quadvf_hist(r, prefix, L = 10, Nmax = 5000, ftol = 1e-9):
	H = mor.demos.build_bg_delay()
	H_norm = H.quad_norm(n = n_quad, L = 10)

	fom_eval_hist = []
	rel_err_hist = []
	for N in range(2*r, Nmax, 10):	
		Hr = mor.QuadVF(r, N = N, L = L, ftol = ftol)
		Hr.fit(H)	
		err = (H - Hr).quad_norm(n = n_quad, L = L)/H_norm
		hist = Hr.history[-1]
		fom_eval_hist.append( hist['total_fom_evals'] + hist['total_fom_der_evals'] + hist['total_linear_solves'] )
		rel_err_hist.append( (H - Hr).quad_norm(n = n_quad, L = 10)/H_norm)
			
		print(N, fom_eval_hist[-1], rel_err_hist[-1])
		
		pgf = PGF()
		pgf.add('fom_evals', np.array(fom_eval_hist))
		pgf.add('rel_err', np.array(rel_err_hist))
		pgf.write(prefix + '_hist_%02d.dat' % r)

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Run ISS example.')
	parser.add_argument('rs', type = int, nargs = '*', default = np.arange(2,52,2), help = "ROM dimensions to run")
	parser.add_argument('--alg', type = str, default = 'PH2', help = 'which algorithm to use')
	parser.add_argument('--hist', action = 'store_true', help = "if specified, recored the convergence history")

	args = parser.parse_args()
	
	rs = np.array(args.rs)
	alg = args.alg.lower()
	hist = args.hist
	ftol = 1e-9

	if alg == 'ph2':
		run_mor(mor.ProjectedH2MOR, rs, 'data/fig_delay_ph2', verbose = 10, ftol = ftol, hist = hist)
	elif alg == 'tfirka':	
		run_mor(mor.TFIRKA, rs, 'data/fig_delay_tfirka', verbose = 10, ftol = ftol, flipping = True, hist = hist)
	elif alg == 'quadvf':
		if not hist:
			run_mor(mor.QuadVF, rs, 'data/fig_delay_quadvf', verbose = 10, N = 200, ftol = ftol, L = 10, hist = hist)
		else:
			run_quadvf_hist(rs[0], 'data/fig_delay_quadvf', L = 10)

