
import os
import sys
from distutils.core import setup
from setuptools import find_packages

setup(name='uqpce',
      version='1.0.0',
      description="Uncertaintity Quantification",
      long_description="""\
""",
      classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
      ],
      keywords='',
      author='Joanna Schmidt and Ben Phillips',
      author_email='benjamin.d.phillips@nasa.gov',
      packages=find_packages(), # look at all the __init__.py files and load them as packages
      include_package_data=True ,
      install_requires=[
        'pyyaml',
        'sympy',
        'mpi4py',
        'openmdao',
        'numpy>=1.9.2',
        'scipy',
        'pep8',
        'parameterized',
      ],
    )