
import os
from setuptools import setup


basedir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(basedir, 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()


setup(
    name='hallbench',
    version=__version__,
    description='Hall bench control application',
    url='https://github.com/lnls-ima/hall-bench-control',
    author='lnls-ima',
    license='MIT License',
    packages=['hallbench'],
    install_requires=[
        'pyvisa',
        'numpy',
        'scipy',
        'pandas',
        'pyqtgraph',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': ['hall-bench-control=hallbench.hallbenchapp:run'],
     },
    zip_safe=False)
