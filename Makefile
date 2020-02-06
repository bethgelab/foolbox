.PHONY: test
test:
	pytest --pdb --ignore=foolbox/tests/models/test_models_tensorflow_eager.py --ignore=foolbox/tests/models/test_models_caffe.py
	pytest --pdb --cov-append foolbox/tests/models/test_models_tensorflow_eager.py

.PHONY: test
testsetup:
	pip3 install --upgrade pip
	pip3 uninstall tensorflow
	pip3 uninstall tensorflow-cpu
	pip3 uninstall tensorflow-gpu
	pip3 uninstall torch torchvision
	pip3 install -r requirements-dev.txt
	pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
	pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

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
	mypy foolbox/tests/

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
