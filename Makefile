.PHONY: test
test:
	pytest --verbose

.PHONY: black
black:
	black .

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
