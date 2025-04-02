.PHONY: init clean lint test build build-dist install publish compile docker-test docker-test-inspect docs
.DEFAULT_GOAL := build


init:
	pip install -r requirements.txt

lint:
	isort ./admet_ai
	black ./admet_ai
	python -m flake8 ./admet_ai

test:
	nose2 -v --log-capture

build:
	python -m build --skip-dependency-check --no-isolation

install:
	pip install dist/*.whl

uninstall:
	pip uninstall admet_ai -y

qed: clean build uninstall install
	pip list | grep "admet_ai"

compile:
	python -m compileall -f ./admet_ai

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -fr coverage/
	rm -fr dist/
	rm -fr build/
	rm -fr admet_ai.egg-info/
	rm -fr admet_ai/admet_ai.egg-info/
	rm -fr joblib_memmap/
	rm -fr .pytest_cache/
	rm -f .coverage
	rm -f .noseids
	rm -f .profile
	rm -fr htmlcov
