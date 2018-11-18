import numpy as np

def marriage_sort(a, b):
	""" Align two vectors so that entry-wise using the stable marriage algorithm

	Implements the Gale-Shapely algorithm [GS62]_ to align two vectors of complex numbers
	such that the mismatch under the permutation is minimized.


	Parameters
	----------
	a: np.array((n,))
		List of numbers
	b: np.array((n,))
		List of numbers

	Returns
	-------
	I: np.array((n,))
		permutation such that a - b[I] is minimized 
	

	References
	----------
	.. [GS62] College Admissions and the Stability of Marriage. D. Gale and L. S. Shapley.,
		The American Mathematical Monthly, Jan 1962, pp. 9-15.

	"""

	a = np.array(a)
	b = np.array(b)
	assert a.shape == b.shape, "inputs a and b must be the same shape"

	# a_engaged[i] = j is means a[i] is engaged to b[j]
	# We set these to -1 if they aren't engaged
	a_engaged = -np.ones(a.shape, dtype = np.int)
	b_engaged = -np.ones(b.shape, dtype = np.int)

	pref = np.array([np.argsort(np.abs(a_ - b)) for a_ in a], dtype = np.float)

	while True:
		# Find a man who hasn't proposed yet
		try:
			i = int(np.argwhere(a_engaged == -1)[0])
		except IndexError:
			# If everyone is engaged, stop
			break
		# Find who i wants to propose to
		j = int(pref[i,np.isfinite(pref[i,:])][0])
		#print i,"propose to", j, "with mismatch", np.abs(a[i] - b[j])
		# we cannot propose to this person again
		pref[i,np.argwhere(pref[i,:] == j)] = np.nan 
		
		if  b_engaged[j] < 0:
			b_engaged[j] = i
			a_engaged[i] = j
		else:
			fiancee = b_engaged[j]
			# If the woman (b[j]) is already engaged, if she prefers a[i] to a[fiancee], switch
			if abs(a[fiancee] - b[j]) > abs(a[i] - b[j]):
				b_engaged[j] = i
				a_engaged[i] = j
				a_engaged[fiancee] = -1
	return a_engaged

def marriage_norm(a, b):
	""" Returns the 2-norm where entries have been permuted to minimize the norm

	Given two (complex) vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`, 
	compute the permutation :math:`\mathcal{I}` such that the 2-norm is minimized,

	.. math:: 

		\min_{\mathcal I}  \| \mathbf{a} - \mathbf{b}[\mathcal{I}] \|_2
	
	and return that norm.  Here we use the stable marriage algorithm as implemented
	in :meth:`h2mor.marriage_sort`. 

	Parameters
	----------
	a: np.array((n,))
		List of numbers
	b: np.array((n,))
		List of numbers

	Returns
	-------
	norm: float
		2-norm of mismatch
	"""
	I = marriage_sort(a, b)
	return np.linalg.norm(a - b[I])

if __name__ == "__main__":
	a = np.arange(11)
	b = np.arange(11)
	I = np.random.permutation(11)
	b = b[I] + 1e-4*np.random.rand(11)
	I2 = marriage_sort(a,b)
	print a
	print b[I2]
