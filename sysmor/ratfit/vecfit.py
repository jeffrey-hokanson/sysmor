import numpy as np

def import_data(X, y, copy = False):
	M = X.shape[0]
	assert y.shape[0] == M, "Input and output don't have the same dimensions"
	if copy:
		return np.copy(X), np.copy(y)
	else:
		return X, y

def vector_fitting(X, y, m, n, maxiter = 100, field = 'complex', rank_one = True, weight = None):
	r"""

	Parameters
	----------
	X: (M,)
		Input coordinates
	y: (M,...)
	"""

	X, y = import_data(X, y)
	# As this only handles the scalar case
	X = X.flatten()
	assert m + 1 >= n, "Vector fitting can only handle degree m >= n - 1"
	if weight is None:
		weight = lambda x: x


	M = X.shape[0]	

