.PHONY: test
test:
	pytest --cov-report term-missing --cov=foolbox.ext.native --verbose --ignore tests/attacks/
	pytest --cov-report term-missing --cov=foolbox.ext.native --cov-append --verbose --backend pytorch --ignore tests/attacks/
	pytest --cov-report term-missing --cov=foolbox.ext.native --cov-append --verbose --backend tensorflow --ignore tests/attacks/
	pytest --cov-report term-missing --cov=foolbox.ext.native --cov-append --verbose --backend jax --ignore tests/attacks/

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
	mypy -m foolbox.ext.native

.PHONY: install
install:
	pip3 install -e .

.PHONY: devsetup
devsetup:
	pre-commit install

.PHONY: build
build:
	python3 setup.py sdist

.PHONY: commit
commit:
	git add foolbox/ext/native/VERSION
	git commit -m 'Version $(shell cat foolbox/ext/native/VERSION)'

.PHONY: release
release: build
	twine upload dist/foolbox-native-$(shell cat foolbox/ext/native/VERSION).tar.gz
