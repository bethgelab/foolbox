from setuptools import setup
from setuptools import find_packages
from os.path import join, dirname
# We need io.open() (Python 3's default open) to specify file encodings
import io

with open(join(dirname(__file__), 'foolbox/VERSION')) as f:
    version = f.read().strip()

try:
    # obtain long description from README
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    readme_path = join(dirname(__file__), 'README.rst')
    with io.open(readme_path, encoding='utf-8') as fr:
        README = fr.read()
except IOError:
    README = ''


install_requires = [
    'numpy',
    'scipy',
    'setuptools',
    'requests',
    'GitPython'
]

tests_require = [
    'pytest',
    'pytest-cov',
]

setup(
    name="foolbox",
    version=version,
    description="Python toolbox to create adversarial examples that fool neural networks",
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Jonas Rauber & Wieland Brendel",
    author_email="opensource@bethgelab.org",
    url="https://github.com/bethgelab/foolbox",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        ':python_version == "2.7"': ['future', 'futures'],
    },
)
