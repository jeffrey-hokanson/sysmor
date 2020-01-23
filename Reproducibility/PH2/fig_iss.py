import numpy as np
from scipy.linalg import svdvals
from mor import ProjectedH2MOR, IRKA, TFIRKA, QuadVF
from mor import cauchy_ldl
from mor.demos import build_iss
from mor.pgf import PGF
import argparse


def run_mor(MOR, rs, prefix, **kwargs):
	rs = np.atleast_1d(rs)

	H = build_iss()
	# Make a SISO system
	H = H[0,0]

	H_norm = H.norm()

	rel_err = np.nan*np.zeros(rs.shape)	
	fom_evals = np.nan*np.zeros(rs.shape)
	
	# Bode plots
	z = 1j*np.logspace(-1, 3, 600)
	Hz = H.transfer(z)

	for i, r in enumerate(rs):
		# Everything is deterministic, but I'm still seeing changes between runs
		# so I'm pinning the random seed to see if this helps
		np.random.seed(0)
			
		Hr = MOR(r, **kwargs)
		Hr.fit(H)
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

		# Now plot history of this iteration
		fom_eval_hist = [hist['total_fom_evals'] + hist['total_fom_der_evals'] + hist['total_linear_solves'] for hist in Hr.history]
		rel_err_hist = [ (hist['Hr'] - H).norm()/H_norm for hist in Hr.history]

		# Condition number
		cond_M = np.zeros(len(Hr.history))
		for i, hist in enumerate(Hr.history):
			try:
				if np.any(hist['mu'].real == 0):
					raise ValueError
				L,d,p = cauchy_ldl(hist['mu'])
				s = svdvals(L @ np.diag(np.sqrt(d)))**2 
				cond_M[i] = np.max(s)/np.min(s)
			except ValueError:
				cond_M[i] = np.nan

		pgf = PGF()
		pgf.add('fom_evals', fom_eval_hist)
		pgf.add('rel_err', rel_err_hist)
		pgf.add('cond_M', cond_M)	
		pgf.write(prefix + '_%02d.dat' % r)

		# Generate Bode plot
		Hrz = Hr.transfer(z)
		pgf = PGF()
		pgf.add('z', z.imag)
		pgf.add('Hz', np.abs(Hz).flatten())
		pgf.add('Hrz', np.abs(Hrz).flatten())
		pgf.add('diff', np.abs(Hz - Hrz).flatten())
		pgf.write(prefix + '_bode_%02d.dat' % r)

def run_quadvf(Ns, r, L, prefix,  **kwargs):
	Ns = np.atleast_1d(Ns)
	
	H = build_iss()
	# Make a SISO system
	H = H[0,0]

	H_norm = H.norm()

	err = np.zeros(Ns.shape)

	for i, N in enumerate(Ns):
		Hr = QuadVF(r, N, L, **kwargs)	
		Hr.fit(H)

		err[i] = (H - Hr).norm()/H_norm
		print("N=%d, err=%8.2e" % (N, err[i]))


	pgf = PGF()
	pgf.add('N', Ns)
	pgf.add('rel_err', err)
	pgf.write(prefix + '_%02d_L%d.dat' % (r, L))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run ISS example.')
	parser.add_argument('rs', type = int, nargs = '*', default = np.arange(2,52,2), help = "ROM dimensions to run")
	parser.add_argument('--alg', type = str, default = 'PH2', help = 'which algorithm to use')
	args = parser.parse_args()

	ftol = 1e-9
	rs = args.rs
	alg = args.alg.lower()

	if alg == 'quadvf':
		run_mor(QuadVF, rs, 'data/fig_iss_quadvf', verbose = 10, N = 100, ftol = ftol, L = 10)
	elif alg == 'ph2':
		run_mor(ProjectedH2MOR, rs, 'data/fig_iss_ph2', verbose = 10, print_norm = True, ftol = ftol)
	elif alg == 'irka':
		run_mor(IRKA, rs, 'data/fig_iss_irka', verbose = True, print_norm = True, ftol = ftol)
	elif alg == 'tfirka':
		run_mor(TFIRKA, rs, 'data/fig_iss_tfirka', verbose = True, print_norm = True, ftol = ftol)
	else:
		print("Unknown algorithm")

	#Ns = np.arange(25, 1000+1,1)
	#Ls = [10]
	#for L in Ls:
	#	run_quadvf(Ns, 28, L, 'data/fig_iss_quadvf')
	

