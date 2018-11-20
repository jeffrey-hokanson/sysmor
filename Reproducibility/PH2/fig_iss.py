import numpy as np
from mor import ProjectedH2MOR, IRKA, TFIRKA, QuadVF
from mor.demos import build_iss
from mor.pgf import PGF


def run_mor(MOR, rs, prefix, **kwargs):
	rs = np.atleast_1d(rs)

	H = build_iss()
	# Make a SISO system
	H = H[0,0]

	H_norm = H.norm()

	rel_err = np.nan*np.zeros(rs.shape)	
	fom_evals = np.nan*np.zeros(rs.shape)
	
	# Bode plots
	z = 1j*np.logspace(-2, 3, 100)
	Hz = H.transfer(z)

	for i, r in enumerate(rs):
		Hr = MOR(r, **kwargs)
		Hr.fit(H)

		err = (H - Hr).norm()/H_norm
		print "%3d :err %6.2e" % (r, err)
		
		# Save information about overall performance
		rel_err[i] = err
		fom_evals[i] = Hr.history[-1]['total_fom_evals']

		pgf = PGF()
		pgf.add('rom_dim',rs)
		pgf.add('fom_evals', fom_evals)
		pgf.add('rel_err', rel_err)
		pgf.write(prefix+'.dat')

		# Now plot history of this iteration
		fom_eval_hist = [hist['total_fom_evals'] + hist['total_fom_der_evals'] + hist['total_linear_solves'] for hist in Hr.history]	
		rel_err_hist = [ (hist['Hr'] - H).norm()/H_norm for hist in Hr.history]
		pgf = PGF()
		pgf.add('fom_evals', fom_eval_hist)
		pgf.add('rel_err', rel_err_hist)	
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

	pgf = PGF()
	pgf.add('N', Ns)
	pgf.add('rel_err', err)
	pgf.write(prefix + '_%02d_L%4.2f.dat' % (r, L))

if __name__ == '__main__':
	ftol = 1e-9
	rs = np.arange(2,50+2,2)
	#run_mor(ProjectedH2MOR, rs, 'data/fig_iss_ph2', verbose = True, print_norm = True, cond_growth = 5, ftol = 1e-9)
	#run_mor(IRKA, rs, 'data/fig_iss_irka', verbose = True, print_norm = True, ftol = ftol)
	#run_mor(TFIRKA, rs, 'data/fig_iss_irka', verbose = True, print_norm = True, ftol = ftol)

#	Ns = np.arange(50, 2000, 10)
#	Ls = [1, 10, 100]
#	for L in Ls:
#		run_quadvf(Ns, 28, L, 'data/fig_iss_quadvf')
	

