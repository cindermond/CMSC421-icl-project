import setuptools
from setuptools import setup

setup(
    name='batcon',
    version='1.0',
    description='mini-batch in-context learning',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'transformers', 'datasets', 'evaluate', 'accelerate', 'sentencepiece', 'protobuf', 'scikit-learn']
    )