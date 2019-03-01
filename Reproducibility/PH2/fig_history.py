import numpy as np
from mor import ProjectedH2MOR
from mor.demos import build_iss
from mor.pgf import PGF


def run_mor(MOR, r = 2, **kwargs):
	H = build_iss()
	H = H[0,0]


	Hr = MOR(r, **kwargs)	
	Hr.fit(H)
	
	# Bode plots
	z = 1j*np.logspace(-1, 3, 600)
	Hz = H.transfer(z)

	for k, hist in enumerate(Hr.history):
		mu = np.array(hist['mu'])
		pgf = PGF()
		pgf.add('mu_real', mu.real)
		pgf.add('mu_imag', mu.imag)
		pgf.write('data/fig_history_mu_%d.dat' % k)

		lam = hist['Hr'].poles()
		pgf = PGF()
		pgf.add('lam_real', lam.real)
		pgf.add('lam_imag', lam.imag)
		pgf.write('data/fig_history_lam_%d.dat' % k)	

		Hrz = Hr.transfer(z)
		pgf = PGF()
		pgf.add('z', z.imag)
		pgf.add('Hz', np.abs(Hz).flatten())
		pgf.add('Hrz', np.abs(Hrz).flatten())
		pgf.add('diff', np.abs(Hz - Hrz).flatten())
		pgf.write('data/fig_history_bode_%d.dat' % k)
		

	
if __name__ == '__main__':
	run_mor(ProjectedH2MOR, r = 4)
	
