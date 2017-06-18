from setuptools import setup
from os.path import join, dirname

with open(join(dirname(__file__), 'foolbox/VERSION')) as f:
    version = f.read().strip()

setup(version=version)
