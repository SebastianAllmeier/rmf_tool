""""""

import setuptools
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='rmf_tool',
   version='',
   description='',
   license="",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='',
   author_email='',
   url="",
   download_url = '',
   packages=['rmf_tool'],
   keywords=[],
   classifiers=[
       "Intended Audience :: Developers",
       "Intended Audience :: Education",
       "Intended Audience :: End Users/Desktop",
       "Programming Language :: Python :: 3",
       "Operating System :: OS Independent",
   ],

   install_requires=[
        'jax',
        'matplotlib',
        'numpy',
        'scipy',
        'sympy',
   ],

   python_requires='>=3.6'
)