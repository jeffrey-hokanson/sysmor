import numpy as np
from system import TransferSystem 

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
	TranferSystem
		Transfer functions system representing the string model.

	References
	----------
	.. [CM09] Transfer functions of distributed parameter systems: a tutorial,
		R. Curtain and K. Morris. Automatica 45 (2009), p. 1101-1116
	
	"""

	H = lambda z: ( z/2.*np.sinh(z) + 2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1)/ (z*(z+epsilon/2)*np.sinh(z) + epsilon*(2*np.cosh(z/2) - 3*np.cosh(z/2)**2 + 1))
	
	Hp = lambda z: -2*z*(6*z*np.sinh(z/2)**2 - 2*z*np.sinh(z/2)*np.sinh(z) + z*np.sinh(z)**2 - 4*z*np.cosh(z) + z*np.sinh(2*z)/np.sinh(z/2) - 12*np.sinh(z/2)**2*np.sinh(z) + 8*np.sinh(z)*np.cosh(z/2) - 8*np.sinh(z))/(eps*z*np.sinh(z) - 6*eps*np.sinh(z/2)**2 + 4*eps*np.cosh(z/2) - 4*eps + 2*z**2*np.sinh(z))**2
	string = TransferSystem(transfer=H, transfer_der=Hp, lim_zH = 0.5)
	return string
