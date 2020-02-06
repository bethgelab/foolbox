.PHONY: test
test:
	pytest --ignore=foolbox/tests/models/test_models_tensorflow_eager.py --ignore=foolbox/tests/models/test_models_caffe.py
	pytest --cov-append foolbox/tests/models/test_models_tensorflow_eager.py

.PHONY: test
testsetup:
	pip3 install --upgrade numpy
	pip3 install --upgrade scipy
	pip3 install -r requirements-dev.txt
	pip3 uninstall tensorflow
	pip3 uninstall tensorflow-cpu
	pip3 install --upgrade tensorflow==1.14
	pip3 uninstall torch torchvision
	pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

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
	twine upload dist/foolbox-$(shell cat foolbox/VERSION).tar.gz

.PHONY: docs
docs:
	cd docs && make html
