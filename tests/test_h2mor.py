import numpy as np
from sysmor import H2MOR
from sysmor.demos import build_string


def test_eval_transfer_complex():
	""" Check that evaluations of the transfer function are recycled
	"""
	# TODO: Check derivative recycling as well

	h2mor = H2MOR(2, real = False)
	H = build_string()

	mu = np.random.uniform(0,1, size = (6,) ) + 1j*np.random.randn(6) 
	h2mor.eval_transfer(H, mu)
	evals = h2mor._total_fom_evals 
	h2mor.eval_transfer(H, mu)
	assert evals == h2mor._total_fom_evals, "Recycling data didn't work"

def test_eval_transfer_real():
	""" Check that evaluations of the transfer function are recycled
	"""

	h2mor = H2MOR(2, real = True)
	H = build_string()

	mu = np.random.uniform(0,1, size = (6,) ) + 1j*np.random.randn(6) 
	h2mor.eval_transfer(H, mu)
	evals = h2mor._total_fom_evals 
	h2mor.eval_transfer(H, mu)
	assert evals == h2mor._total_fom_evals, "Recycling data didn't work"
	h2mor.eval_transfer(H, mu.conj())
	assert evals == h2mor._total_fom_evals, "Recycling data didn't work"
	
	

