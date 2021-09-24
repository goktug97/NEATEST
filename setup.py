#!/usr/bin/env python

import os
from setuptools import setup

from neatest import VERSION

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='nesneat',
      version=f'{VERSION.major}.{VERSION.minor}.{VERSION.patch}',
      description='Evolving Neural Networks through Augmenting Topologies with'
      'Evolution Strategy Training',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['neatest'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=[
          'numpy',
          'cloudpickle'
      ],
      python_requires='>=3.6',
      include_package_data=True)
