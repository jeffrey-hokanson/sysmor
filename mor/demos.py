import numpy as np
import scipy.sparse as ss
from .system import TransferSystem, StateSpaceSystem 
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

	Note: according to [MorXX]_ p. 23, this is in :math:`\mathcal{H}_\infty`, not :math:`\mathcal{H}_2`.


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

	.. [MorXX] http://www.math.uwaterloo.ca/~kmorris/Preprints/Morris_controlhandbook.pdf	
	"""

	H = lambda z: ( z/2.*np.sinh(z) + 2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1)/ (z*(z+epsilon/2)*np.sinh(z) + epsilon*(2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1))
	
	Hp = lambda z: -2*z*(6*z*np.sinh(z/2)**2 - 2*z*np.sinh(z/2)*np.sinh(z) + z*np.sinh(z)**2 - 4*z*np.cosh(z) + z*np.sinh(2*z)/np.sinh(z/2) - 12*np.sinh(z/2)**2*np.sinh(z) + 8*np.sinh(z)*np.cosh(z/2) - 8*np.sinh(z))/(eps*z*np.sinh(z) - 6*eps*np.sinh(z/2)**2 + 4*eps*np.cosh(z/2) - 4*eps + 2*z**2*np.sinh(z))**2
	string = TransferSystem(transfer=H, transfer_der=Hp, lim_zH = (0.5,0.5), isreal = True, vectorized = True)
	return string


def build_beam2():
	r""" Clamped beam with Kelvin-Voigt damping

	This function implements the clamped beam example from subsection 4.2 of [CM09]_.

	Here we have set all the parameters to one:

	.. math::

		E = I = c_d = L = 1
	


	References
	----------
	.. [CM09] Transfer functions of distributed parameter systems: a tutorial,
		R. Curtain and K. Morris. Automatica 45 (2009), p. 1101-1116

	"""
	# TODO: Eventually allow all the parameters to be set
	# TODO: Implement tests of this implementation
	def H(z):
		m = (-z**2/(z+1))**(1/4)
		
		N = np.cosh(m)*np.sin(m) - np.sinh(m)*np.cos(m)
		D = 1 + np.cosh(m)*np.cos(m)
		return (z*N)/( m**3*(1+ z)*D)

	def Hp(z):
		m = (-z**2/(z+1))**(1/4)
		mp = m*(0.25*z+0.5)/(z*(z+1))
	
		N = np.cosh(m)*np.sin(m) - np.sinh(m)*np.cos(m)
		Np = 2*np.sin(m)*np.sinh(m)*mp
		
		D = 1 + np.cosh(m)*np.cos(m)
		Dp = -(np.sin(m)*np.cosh(m) - np.cos(m)*np.sinh(m))*mp

		Hp = -3*z*(z+1)*D*N*mp - z*(z+1)*N*m*Dp -z*D*N*m + (z+1)*(z*Np+N)*D*m
		Hp *= 1./( (z+1)**2 * D**2 * m**4)	
		return Hp

	# Note we do not define lim_zH as lim_{omega\to\infty} omega*H(1j*omega) diverges

	beam = TransferSystem(transfer = H, transfer_der = Hp, 
			isreal = True, vectorized = True)

	return beam


def build_beam6():
	r""" Pinned-free beam with shear force control

	This function implements the pinned-free beam example from subsection 4.4 of [CM09]_.

	Here we have set all the parameters to one:

	.. math::

		E = I = c_d = L = 1
	


	References
	----------
	.. [CM09] Transfer functions of distributed parameter systems: a tutorial,
		R. Curtain and K. Morris. Automatica 45 (2009), p. 1101-1116

	"""

	def H(z):
		m = (-z**2/(z+1))**(1/4)
		N = np.cosh(m)*np.sin(m) - np.sinh(m)*np.cos(m)
		
		H = -2*np.sinh(m)*np.sin(m)/( m**3 * (1+z)*N)
		return H

#	def Hp(z):
#		m = (-z**2/(z+1))**(1/4)
#		mp = m*(0.25*z+0.5)/(z*(z+1))
#		
#		N = np.cosh(m)*np.sin(m) - np.sinh(m)*np.cos(m)
#		Np = 2*np.sin(m)*np.sinh(m)*mp

	beam = TransferSystem(transfer = H, isreal = True, vectorized = True)
	return beam

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


def build_subg_delay(n = 500, tau = 1, rho = 5, eps = 0.01):
	r""" Delay Differential Equation Example of Beattie and Gugercin


	.. math::
		


	"""

	I = ss.eye(n)
	
	# Serkan's Matlab code
	# T=diag(ones(dimN-1,1),1)+diag(ones(dimN-1,1),-1); T(1,1)=1; T(dimN,dimN)=1;
	T = ss.diags(np.ones((2,n-1)), [1,-1], shape = (n,n))
	data = np.array([1,1])
	row = np.array([0,n-1])
	col = np.array([0,n-1])
	T += ss.coo_matrix(( data, (row, col)), shape = (n,n)) 

	
	# Serkan's Matlab code
	#E=(2/sqrt(1-epsilon))*eye(dimN,dimN) + T;
	#E = 2/np.sqrt(1-eps)*I + T
	#A0= (-1/sqrt(1-epsilon))*(2/tau)*(1+(1/rho))*eye(dimN,dimN) + (1/tau)*(1+(1/rho))*T;
	#A0 = (-1/np.sqrt(1-eps))*(2/tau)*(1+(1/rho))*I + (1/tau)*(1+(1/rho))*T
	#A1=(-1/sqrt(1-epsilon))*(2/tau)*(-1+(1/rho))*eye(dimN,dimN) + (1/tau)*(-1+(1/rho))*T;
	#A1 = (-1/np.sqrt(1-eps))*(2/tau)*(-1+(1/rho))*I + (1/tau)*(-1+(1/rho))*T

	A1 = rho*I + T
	A2 = 1/tau*(1/eps + 1)*(T - rho*I)
	A3 = 1/tau*(1/eps - 1)*(T - rho*I)

	B = np.zeros((n,1))
	B[0,0] = 1
	B[1,0] = 1
	C = np.zeros((1,n))
	C[0,0] = 1
	C[0,1] = 1
	#H = lambda z: C @ ss.linalg.spsolve( z * E - A0 - np.exp(-tau*z)* A1, B)
	#Hder = lambda z: - C @ ss.linalg.spsolve( z * E - A0 - np.exp(-tau*z)* A1, 
	#						(E + tau * np.exp(-tau*z)*A1) @ ss.linalg.spsolve( z*E - A0 - np.exp(-tau*z)*A1, B))
	
	H = lambda z: C @ ss.linalg.spsolve( z * A1 + A2 +  np.exp(-tau*z) * A3, B)
	Hder = lambda z: - C @ ss.linalg.spsolve( z * A1 + A2 + np.exp(-tau*z) * A3, 
							(A1 + tau * np.exp(-tau*z)*A3) @ ss.linalg.spsolve( z * A1 + A2 + np.exp(-tau*z) * A3, B))

	
	# Limits 
	lim_zH = C @ ss.linalg.spsolve(A1, B)	
	sys = TransferSystem(H, transfer_der = Hder, isreal = True, lim_zH = [lim_zH, lim_zH] )
	sys.A1 = A1
	sys.A2 = A2
	sys.A3 = A3
	sys.tau = tau
	sys.eps = eps
	sys.rho = rho
	return sys

def build_bg_delay(n = 1000, tau = 1, rho = 0.1, eps = 0.99):
	r""" Delay Differential Equation Example of Beattie and Gugercin


	.. math::
		


	"""

	I = ss.eye(n)
	
	# Serkan's Matlab code
	# T=diag(ones(dimN-1,1),1)+diag(ones(dimN-1,1),-1); T(1,1)=1; T(dimN,dimN)=1;
	T = ss.diags(np.ones((2,n-1)), [1,-1], shape = (n,n))
	data = np.array([1,1])
	row = np.array([0,n-1])
	col = np.array([0,n-1])
	T += ss.coo_matrix(( data, (row, col)), shape = (n,n)) 

	
	# Serkan's Matlab code
	#E=(2/sqrt(1-epsilon))*eye(dimN,dimN) + T;
	E = 2/np.sqrt(1-eps)*I + T
	#A0= (-1/sqrt(1-epsilon))*(2/tau)*(1+(1/rho))*eye(dimN,dimN) + (1/tau)*(1+(1/rho))*T;
	A0 = (-1/np.sqrt(1-eps))*(2/tau)*(1+(1/rho))*I + (1/tau)*(1+(1/rho))*T
	#A1=(-1/sqrt(1-epsilon))*(2/tau)*(-1+(1/rho))*eye(dimN,dimN) + (1/tau)*(-1+(1/rho))*T;
	A1 = (-1/np.sqrt(1-eps))*(2/tau)*(-1+(1/rho))*I + (1/tau)*(-1+(1/rho))*T

	B = np.zeros((n,1))
	B[0,0] = 1
	B[1,0] = 1
	C = np.zeros((1,n))
	C[0,0] = 1
	C[0,1] = 1
	H = lambda z: C @ ss.linalg.spsolve( z * E - A0 - np.exp(-tau*z)* A1, B)
	Hder = lambda z: - C @ ss.linalg.spsolve( z * E - A0 - np.exp(-tau*z)* A1, 
							(E + tau * np.exp(-tau*z)*A1) @ ss.linalg.spsolve( z*E - A0 - np.exp(-tau*z)*A1, B))
	
	
	# Limits 
	lim_zH = C @ ss.linalg.spsolve(E, B)	
	sys = TransferSystem(H, transfer_der = Hder, isreal = True, lim_zH = [lim_zH, lim_zH] )
	sys.E = E
	sys.A0 = A0
	sys.A1 = A1
	sys.tau = tau
	sys.eps = eps
	sys.rho = rho
	return sys
