import os
from setuptools import setup


with open('README.md', 'r') as f:
	long_description = f.read()

setup(name='sysmor',
	version = '0.1',
	url = 'https://github.com/jeffrey-hokanson/sysmor',
	description = 'System-Theoertic Model Order Reduction',
	long_description = long_description,
	long_description_content_type = 'text/markdown', 
	author = 'Jeffrey M. Hokanson',
	author_email = 'jeffrey@hokanson.us',
	packages = ['sysmor',],
	install_requires = [
		'numpy', 
		'scipy', 
		'matplotlib',
		'mpmath',
		'polyrat',
		'iterprinter',
		],
	python_requires='>=3.6',
	)
