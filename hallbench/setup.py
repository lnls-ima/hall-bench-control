#!/usr/bin/env python3

from setuptools import setup

with open('VERSION','r') as _f:
    __version__ = _f.read().strip()

setup(
    name='hallbench',
    version=__version__,
    author='lnls-ima',
    description='Hall Bench Package',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=['hallbench'],
    package_data={'hallbench': ['VERSION']},
    zip_safe=False
)
