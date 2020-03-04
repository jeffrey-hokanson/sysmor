import os
from setuptools import setup


setup(name='sysmor',
	version = '0.1',
	description = 'System-Theoertic Model Order Reduction',
	author = 'Jeffrey M. Hokanson',
	packages = ['sysmor',],
	install_requires = [
		'numpy', 
		'scipy', 
		'matplotlib',
		'mpmath',
		],
	)
