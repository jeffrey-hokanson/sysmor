import os
from setuptools import setup


setup(name='mor',
	version = '0.1',
	description = 'Model Order Reduction',
	author = 'Jeffrey M. Hokanson',
	packages = ['mor', 'mor.opt'],
	install_requires = [
		'numpy', 
		'scipy', 
		'matplotlib<=2.2.3'
		],
	)
