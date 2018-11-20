import numpy as np
from system import TransferSystem, StateSpaceSystem 
import os 

# Extract the path of this file
import scipy.io
dir_path = os.path.dirname(os.path.realpath(__file__))

def build_string(epsilon = 1):
	""" String equation model damped on half the interval

	This function builds a transfer function defined system based on 
	a string model from [CM09]_ subsection 2.2 as given by

	.. math::

		H(z) = \\frac{
			\\frac{z}{2} \sinh(z) + 2\cosh(\\frac{z}{2}) -3\cosh^2(\\frac{z}{2}) + 1
		}{
			z(z+\\frac{\epsilon}{2})\sinh(z) + \epsilon(2\cosh(\\frac{z}{2}) - 3\cosh^2(\\frac{z}{2})+1)
		}.

	Parameters
	----------
	epsilon: float, optional
		positive scalar damping parameter; defaults to 1

	Returns
	-------
	TransferSystem
		Transfer functions system representing the string model.

	References
	----------
	.. [CM09] Transfer functions of distributed parameter systems: a tutorial,
		R. Curtain and K. Morris. Automatica 45 (2009), p. 1101-1116
	
	"""

	H = lambda z: ( z/2.*np.sinh(z) + 2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1)/ (z*(z+epsilon/2)*np.sinh(z) + epsilon*(2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1))
	
	Hp = lambda z: -2*z*(6*z*np.sinh(z/2)**2 - 2*z*np.sinh(z/2)*np.sinh(z) + z*np.sinh(z)**2 - 4*z*np.cosh(z) + z*np.sinh(2*z)/np.sinh(z/2) - 12*np.sinh(z/2)**2*np.sinh(z) + 8*np.sinh(z)*np.cosh(z/2) - 8*np.sinh(z))/(eps*z*np.sinh(z) - 6*eps*np.sinh(z/2)**2 + 4*eps*np.cosh(z/2) - 4*eps + 2*z**2*np.sinh(z))**2
	string = TransferSystem(transfer=H, transfer_der=Hp, lim_zH = (0.5,0.5), isreal = True)
	return string


def build_cdplayer(sparse = False):
	r""" The CD Player model

	This is the famous CD player model with two inputs and two outputs
	described in [CD02]_.
	

	Parameters
	----------
	sparse: bool
		If true, construct a sparse state-space system

	Returns
	-------
	StateSpaceSystem or SparseStateSpaceSystem

	References
	----------
	.. [CD02] Younes Chahlaoui and Paul Van Dooren.
		A collection of benchmark examples for model reduction of linear time invariant dynamical systems; 
		SLICOT Working Note 2002-2: February 2002.
	"""
	fname = os.path.join(dir_path, 'data/CDPlayer.mat')
	data = scipy.io.loadmat(fname)
	A = data['A']
	B = data['B']
	C = data['C']

	if sparse:
		raise NotImplementedError

	return StateSpaceSystem(A, B, C)

def build_iss(sparse = False):
	r""" Model of the International Space Station 1r component

	
	This model of the space station is described in [CD02]_.
	

	Parameters
	----------
	sparse: bool
		If true, construct a sparse state-space system

	Returns
	-------
	StateSpaceSystem or SparseStateSpaceSystem

	References
	----------
	.. [CD02] Younes Chahlaoui and Paul Van Dooren.
		A collection of benchmark examples for model reduction of linear time invariant dynamical systems; 
		SLICOT Working Note 2002-2: February 2002.
	"""
	fname = os.path.join(dir_path, 'data/iss.mat')
	data = scipy.io.loadmat(fname)
	A = data['A']
	B = data['B']
	C = data['C']

	if sparse:
		raise NotImplementedError

	return StateSpaceSystem(A, B, C)

