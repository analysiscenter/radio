"""
RadIO is a framework for batch-processing of computational tomography (CT)-scans
for deep learning experiments.
Documentation - https://analysiscenter.github.io/radio/
"""

from setuptools import setup, find_packages
import re

with open('radio/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('docs/source/index.rst', 'r') as f:
    long_description = f.read()


setup(
    name='radio',
    packages=find_packages(exclude=['examples']),
    version=version,
    url='https://github.com/analysiscenter/radio',
    license='Apache License 2.0',
    author='Data Analysis Center team',
    author_email='radio@analysiscenter.ru',
    description='A framework for deep research of CT scans',
    long_description=long_description,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.11',
        'scipy>=1.17.0',
        'scikit-build>=0.5.0',
        'Cython>=0.23',
        'matplotlib>=1.3.1',
        'networkx>=1.8',
        'six>=1.10.0',
        'pillow>=2.1.0',
        'PyWavelets>=0.4.0',
        'dask[array]>=0.9.0',
        'wfdb>=1.2.2.',
        'dill>=0.2.7.1',
        'scikit-learn',
        'numba>=0.36.1',
        'blosc>=1.5.0',
        'scikit-image>=0.13.0',
        'cloudpickle>=0.5.2',
        'pydicom>=0.9.9',
        'aiofiles>=0.3.1',
        'SimpleITK'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.4'],
        'keras': ['keras>=2.0.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)
