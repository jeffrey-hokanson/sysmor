import numpy as np
import scipy
from itertools import product

from iterprinter import IterationPrinter
from .system import DescriptorSystem, DiagonalStateSpaceSystem
from .marriage import hungarian_norm

def tangential_hermite_interpolant(z, Hb, cH, cHpb, b, c):
	r"""

	From BG12, eqs. 10-12 
	"""
	
	r = len(z)
	p = len(b[0])
	m = len(c[0])
	
	A = np.zeros((r, r), dtype = complex)
	E = np.zeros((r, r), dtype = complex)

	for i, j in product(range(r), range(r)):
		if i != j:
			cHbi = cH[i] @ b[j]	# c[i] @ H[i] @ b[j]
			cHbj = c[i] @ Hb[j]	# c[i] @ H[j] @ b[j]
			A[i,j] = - (z[i]*cHbi - z[j]*cHbj)/(z[i] - z[j])
			E[i,j] = - (cHbi - cHbj)/(z[i] - z[j])
		else:
			A[i,j] = - c[i] @ Hb[i] - z[i]*cHpb[i]
			E[i,j] = - cHpb[i]

	B = np.zeros((r, p), dtype = complex)
	C = np.zeros((m, r), dtype = complex)
	
	for i in range(r):
		B[i,:] = cH[i].reshape(p)
		C[:,i] = Hb[i].reshape(m)

	return DescriptorSystem(A, B, C, E)	

def modal_truncation(H, r, which = 'LR'):
	ew, evR = H.eig(which = which, k = r, right = True)
	C = H.C @ evR
	B = scipy.linalg.lstsq(evR, H.B)[0]
	return DiagonalStateSpaceSystem(ew, B, C)

def residue_correction(H, Hr, maxiter = 10, pinv_tol = 5e-7, tol = 2e-7, verbose = True):

	if verbose:
		printer = IterationPrinter(it = '4d', B_norm = '10.3e', C_norm = '10.3e')
		printer.print_header(it = 'iter', B_norm = 'Δ B', C_norm = 'Δ C')

	Hr = Hr.to_diagonal()

	lam = Hr.poles()

	Y = np.zeros((Hr.state_dim, Hr.output_dim), dtype = complex)
	Z = np.zeros((Hr.state_dim, Hr.input_dim), dtype = complex) 
	M = np.zeros((Hr.state_dim, Hr.state_dim), dtype = complex)

	B = Hr.B
	C = Hr.C
	for it in range(maxiter):
		C_old = np.copy(C)
		B_old = np.copy(B)

		##############################
		# Update C matrix
		##############################	

		# Form data
		for i in range(Hr.state_dim):
			Y[i] = H.transfer(lam[i], right_tangent = B[i]).flatten()

		# Evaluate Gram matrix (Cauchy)
		for i, j in product(range(Hr.state_dim), range(Hr.state_dim)):
			M[i,j] = (B[i].conj().T @ B[j])/(-lam[i].conj() - lam[j])

		# rank-truncated pseudo-inverse
		sig, U = scipy.linalg.eigh(M)
		I = sig>pinv_tol
		C = Y.T.conj() @ U[:,I].conj() @ np.diag(1./sig[I]) @ U[:,I].T
		
		##############################
		# Update B matrix
		##############################	
	
		# Form data
		for i in range(Hr.state_dim):
			Z[i] = H.transfer(lam[i], left_tangent = C[:,i]).flatten()
		
		# Evaluate Gram matrix (Cauchy)
		for i, j in product(range(Hr.state_dim), range(Hr.state_dim)):
			M[i,j] = (C[:,i].conj().T @ C[:,j])/(-lam[i].conj() - lam[j])

		# rank-truncated pseudo-inverse
		sig, U = scipy.linalg.eigh(M)
		I = sig>pinv_tol
		B = (Z.T.conj() @ U[:,I].conj() @ np.diag(1./sig[I]) @ U[:,I].T).T
		
		############################
		# Balance
		############################
		for i in range(Hr.state_dim):
			tau = np.sqrt(np.linalg.norm(C[:,i])/np.linalg.norm(B[i]))
			C[:,i] *= 1./tau
			B[i,:] *= tau
		
		# Check termination 
		B_norm = np.linalg.norm( (B - B_old).flatten(), np.inf)
		C_norm = np.linalg.norm( (C - C_old).flatten(), np.inf)

		if verbose:
			printer.print_iter(it = it + 1, B_norm = B_norm, C_norm = C_norm)

		if B_norm < tol and C_norm < tol:
			break 

	return DiagonalStateSpaceSystem(lam, B, C)


def tfirka(H, rom_dim, Hr0 = None, maxiter = 100, verbose = True, ztol = 1e-7, flip = True, 
	residue_correct = True):

	if Hr0 is None:
		Hr0 = modal_truncation(H, rom_dim)

	if verbose:
		printer = IterationPrinter(it = '4d', delta_z = '10.3e', flip = '^5s')
		printer.print_header(it = 'iter', delta_z = 'Δ poles', flip = 'flip?')
	
	Hr = Hr0.to_diagonal()

	# Storage for constructing interpolant
	Hb = np.zeros((rom_dim, H.output_dim), dtype = complex)	
	cH = np.zeros((rom_dim, H.input_dim), dtype = complex)
	cHpb = np.zeros((rom_dim,), dtype = complex)

	flipped = 'N'

	for it in range(maxiter):
		
		z = -Hr.ew.conj()
		if flip:
			z_new = np.abs(z.real) + 1j*z.imag
			if not np.allclose(z_new, z):
				flipped = 'Y'
			else:
				flipped = 'N'

		for i in range(rom_dim):
			Hb_, Hpb_ = H.transfer(z[i], right_tangent = Hr.B[i], der = True)
			Hb[i] = Hb_.flatten()
			cH_ = H.transfer(z[i], left_tangent = Hr.C[:,i])
			cH[i] = cH_.flatten()
			cHpb[i] = (Hr.C[:,i] @ Hpb_).flatten()

		# Build interpolant	
		Hr = tangential_hermite_interpolant(z, Hb, cH, cHpb, Hr.B, Hr.C.T)
		
		# Residue correct
		if residue_correct:
			Hr = residue_correction(H, Hr)		
	
		Hr = Hr.to_diagonal()
		z_new = -Hr.ew.conj()
	
		#print(z_new[np.argsort(z_new.imag)])	
		delta_z = hungarian_norm(z, z_new)
		if verbose:
			printer.print_iter(it = it+1, delta_z = delta_z, flip = flipped) 

		if delta_z < ztol:
			break

	return Hr
