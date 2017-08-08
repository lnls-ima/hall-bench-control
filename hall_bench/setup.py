#!/usr/bin/env python3

from setuptools import setup

with open('VERSION', 'r') as _f:
    __version__ = _f.read().strip()

setup(
    name='hall_bench',
    version=__version__,
    author='lnls-ima',
    description='Hall Bench Package',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=['hall_bench'],
    package_data={'hall_bench': ['VERSION']},
    zip_safe=False
)
