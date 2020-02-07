.PHONY: test
test:
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --verbose
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend pytorch
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend jax
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend theano
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend mxnet
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend keras
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend tensorflow-eager
	pytest --pdb --cov-report term-missing --cov=foolbox --doctest-modules --cov-append --verbose --backend tensorflow-graph

.PHONY: black
black:
	black .

.PHONY: blackcheck
blackcheck:
	black --check .

.PHONY: flake8
flake8:
	flake8

.PHONY: mypy
mypy:
	mypy -p foolbox
	mypy tests/

.PHONY: install
install:
	pip3 install -e .

.PHONY: devsetup
devsetup:
	pre-commit install

.PHONY: build
build:
	python3 setup.py sdist

.PHONY: release
release: build
	pip3 install twine==3.1.1
	twine upload dist/foolbox-$(shell cat foolbox/VERSION).tar.gz

.PHONY: docs
docs:
	cd docs && make html
