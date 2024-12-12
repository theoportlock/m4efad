#!/usr/bin/env python
from setuptools import setup, find_packages
import os

# Load requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Load scripts
scripts = [os.path.join('metatoolkit', f) for f in os.listdir('metatoolkit')
         if (os.path.isfile(os.path.join('metatoolkit', f))) & (f[0] != '_')]

setup(
    name='metatoolkit',
    version='0.2',
    description='A toolkit for meta-analysis and data processing',
    author='Theo Portlock',
    author_email='theo.portlock@auckland.ac.nz',
    url='https://github.com/theoportlock/metatoolkit',  # Optional
    packages=find_packages(),
    scripts=scripts,
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
