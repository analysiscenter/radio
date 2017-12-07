"""
Dataset helps you conveniently work with random or sequential batches of your data
and define data processing and machine learning workflows even for datasets that do not fit into memory.

Documentation - https://analysiscenter.github.io/dataset/
"""
from setuptools import setup, find_packages
import re

with open('dataset/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('docs/index.rst', 'r') as f:
    long_description = f.read()


setup(
    name='dataset',
    packages=find_packages(exclude=['examples']),
    version=version,
    url='https://github.com/analysiscenter/dataset',
    license='Apache License 2.0',
    author='Roman Kh at al',
    author_email='rhudor@gmail.com',
    description='A framework for fast data processing and ML models training',
    long_description=long_description,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.10',
        'dill>=0.2.7',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.13'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.13'],
        'keras': ['keras>=2.0.0']
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
