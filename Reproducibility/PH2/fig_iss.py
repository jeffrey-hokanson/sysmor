import numpy as np
from mor import ProjectedH2MOR
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
		fom_eval_hist = [hist['total_fom_evals'] + hist['total_fom_der_evals'] + hist['total_lin_solve'] for hist in Hr.history]	
		rel_err_hist = [ (hist['Hr'] - H).norm()/H_norm for hist in Hr.history]
		pgf = PGF()
		pgf.add('fom_evals', fom_eval_hist)
		pgf.add('rel_err', rel_err_hist)	
		pgf.write(prefix + '_%02d.dat' % r)

if __name__ == '__main__':
	rs = np.arange(2,10+2,2)
	run_mor(ProjectedH2MOR, rs, 'data/fig_iss_ph2', verbose = True, print_norm = True, cond_growth = 5, ftol = 1e-9)
