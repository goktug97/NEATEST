#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='NEAT',
      version='0.0.1',
      description='Python Implementation of NEAT Genetic Algorithm',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/goktug97/NEAT',
      download_url=(
          'https://github.com/goktug97/NEAT/archive/v0.0.1.tar.gz'),
      py_modules=[os.path.splitext(os.path.basename(path))[0]
                  for path in ['neat.py', 'genome.py', 'connection.py', 'node.py']],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: POSIX :: Linux",
      ],
      python_requires='>=3.6',
      include_package_data=True)
