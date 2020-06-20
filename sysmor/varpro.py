r""" Utility library for using Variable Projection (VARPRO)

This provides a numerically careful implementation of the computation of the 
VARPRO residual and Jacobian based on the discussion in Higham 2002, Sec~19.4

"""
import numpy as np
import scipy.linalg as sl

def inv_perm(p):
	pinv = np.zeros(p.shape, dtype = p.dtype)
	pinv[p] = np.arange(len(p))
	return pinv

def varpro_residual_jacobian_colwise(b, A, DA = None):
	r""" Form the residual and optionally the Jacobian

	Computes the residual vector and optionally the Jacobian for the
	orthogonal projector perpendicular to the subspace spanned by A,
	denoted P_A multiplied by b
	
	..math::

		r = P_A^\perp b.

	This implementation assumes that each column of A depends on a single variable
	stored in the columns of DA; i.e., the derivative A with respect to the $k$ column
	is a matrix that is zero except the kth column which is DA[:,k].

	Parameters
	---------- 
	"""

	# Based Higham 2002, Thm. 19.6, we want to both
	# sort the rows of A to have decreasing sup-norm
	# and then perform QR with column pivoting

	# Sort rows of A
	row_norm = np.max(np.abs(A), axis = 1)
	row_perm = np.argsort(-row_norm)

	# QR with column permutation
	Q, R, col_perm = sl.qr(A[row_perm, :], mode = 'economic', pivoting = True)
	
	QHb = Q.T.conj() @ b[row_perm]

	# Residual
	r = b[row_perm] - Q @ QHb 

	row_perm_inv = inv_perm(row_perm)

	if DA is None:
		return r[row_perm_inv]


	# Compute the linear coefficients 
	# r = b - A @ a
	try:
		a = sl.solve_triangular(R, QHb)
	except sl.LinAlgError as e:
		print("solve_triangular failed")
		raise e
		
	# Form the first term in the Jacobian
	K = np.multiply(DA[row_perm][:,col_perm], a)
	K -= Q @ Q.T.conj() @ K

	# Second term in the Jacobian
	L = -Q @ sl.solve_triangular(R, np.diag(DA.conj().T @ r), trans = 'C')
	
	col_perm_inv = inv_perm(col_perm)

	return r[row_perm_inv], K[row_perm_inv][:,col_perm_inv], L[row_perm_inv][:,col_perm_inv]

