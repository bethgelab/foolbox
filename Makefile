.PHONY: test
test:
	pytest --pdb --cov=foolbox.ext.native
	pytest --pdb --cov=foolbox.ext.native --cov-append --backend pytorch
	pytest --pdb --cov=foolbox.ext.native --cov-append --backend tensorflow
	pytest --pdb --cov=foolbox.ext.native --cov-append --backend jax
	pytest --pdb --cov=foolbox.ext.native --cov-append --backend numpy

.PHONY: test
testskipslow:
	pytest --pdb --skipslow --cov=foolbox.ext.native
	pytest --pdb --skipslow --cov=foolbox.ext.native --cov-append --backend pytorch
	pytest --pdb --skipslow --cov=foolbox.ext.native --cov-append --backend tensorflow
	pytest --pdb --skipslow --cov=foolbox.ext.native --cov-append --backend jax
	pytest --pdb --skipslow --cov=foolbox.ext.native --cov-append --backend numpy

.PHONY: testattacks
testattacks:
	pytest --pdb --cov=foolbox.ext.native.attacks tests/test_attacks.py tests/attacks/
	pytest --pdb --cov=foolbox.ext.native.attacks tests/test_attacks.py tests/attacks/ --cov-append --backend pytorch
	pytest --pdb --cov=foolbox.ext.native.attacks tests/test_attacks.py tests/attacks/ --cov-append --backend tensorflow
	pytest --pdb --cov=foolbox.ext.native.attacks tests/test_attacks.py tests/attacks/ --cov-append --backend jax
	pytest --pdb --cov=foolbox.ext.native.attacks tests/test_attacks.py tests/attacks/ --cov-append --backend numpy

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
	mypy -p foolbox.ext.native
	mypy tests/
	# mypy tests/attacks/

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
