from setuptools import setup
from os.path import join, dirname

with open(join(dirname(__file__), "foolbox/ext/native/VERSION")) as f:
    version = f.read().strip()

try:
    # obtain long description from README
    readme_path = join(dirname(__file__), "README.rst")
    with open(readme_path, encoding="utf-8") as f:
        README = f.read()
except IOError:
    README = ""


install_requires = ["numpy", "scipy", "setuptools", "foolbox>=2.2.1", "eagerpy==0.2.2"]
tests_require = ["pytest"]


setup(
    name="foolbox-native",
    version=version,
    description="Foolbox Native is an extension for Foolbox that tries to bring native performance to Foolbox. This extension is a prototype with the goal of ultimately becoming part of Foolbox itself.",
    long_description=README,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Jonas Rauber",
    author_email="git@jonasrauber.de",
    url="https://github.com/jonasrauber/foolbox-native",
    license="",
    packages=["foolbox.ext", "foolbox.ext.native"],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={"testing": tests_require},
)
