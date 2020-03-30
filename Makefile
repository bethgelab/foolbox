.PHONY: test
test:
	pytest --pdb --cov=foolbox
	pytest --pdb --cov=foolbox --cov-append --backend pytorch
	pytest --pdb --cov=foolbox --cov-append --backend tensorflow
	pytest --pdb --cov=foolbox --cov-append --backend jax
	pytest --pdb --cov=foolbox --cov-append --backend numpy

.PHONY: testskipslow
testskipslow:
	pytest --pdb --skipslow --cov=foolbox
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend pytorch
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend tensorflow
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend jax
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend numpy

.PHONY: testskipslowrev
testskipslowrev:
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend numpy
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend jax
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend tensorflow
	pytest --pdb --skipslow --cov=foolbox --cov-append --backend pytorch
	pytest --pdb --skipslow --cov=foolbox

.PHONY: testattacks
testattacks:
	pytest --pdb --cov=foolbox.attacks tests/test_attacks.py
	pytest --pdb --cov=foolbox.attacks tests/test_attacks.py --cov-append --backend pytorch
	pytest --pdb --cov=foolbox.attacks tests/test_attacks.py --cov-append --backend tensorflow
	pytest --pdb --cov=foolbox.attacks tests/test_attacks.py --cov-append --backend jax
	pytest --pdb --cov=foolbox.attacks tests/test_attacks.py --cov-append --backend numpy

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
	mypy examples/

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
	git add foolbox/VERSION
	git commit -m 'Version $(shell cat foolbox/VERSION)'

.PHONY: release
release: build
	twine upload dist/foolbox-$(shell cat foolbox/VERSION).tar.gz

.PHONY: guide
guide:
	cd guide && vuepress build --temp /tmp/

.PHONY: installvuepress
installvuepress:
	curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
	echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
	sudo apt update && sudo apt install yarn
	sudo yarn global add vuepress

.PHONY: serveguide
serveguide:
	cd guide/.vuepress/dist/ && python3 -m http.server 9999

.PHONY: devguide
devguide:
	cd guide && vuepress dev --temp /tmp/ --port 9999

.PHONY: pushguide
pushguide:
	cd guide/.vuepress/dist/ && git init && git add -A && git commit -m 'deploy'
	cd guide/.vuepress/dist/ && git push -f git@github.com:bethgelab/foolbox.git master:gh-pages
