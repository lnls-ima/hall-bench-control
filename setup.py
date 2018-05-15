
from setuptools import setup

setup(
    name='hallbench',
    version='0.1.0',
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
