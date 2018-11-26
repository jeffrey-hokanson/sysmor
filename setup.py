import os
from setuptools import setup


setup(name='mor',
	version = '0.1',
	description = 'Model Order Reduction',
	author = 'Jeffrey M. Hokanson',
	packages = ['mor', 'mor.opt'],
	install_requires = [
		'numpy', 
		'scipy<=1.1.0', 
		'matplotlib<=2.2.3',
		'mpmath',
		'functools32',
		],
	)
