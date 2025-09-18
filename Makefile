.DEFAULT_GOAL := build
BUCKET  ?= rocketvector-packages

.PHONY: init
init:
	pip install -r requirements.txt

.PHONY: lint 
lint:
	isort ./admet_ai
	black ./admet_ai
	ruff check ./admet_ai --fix

.PHONY: test
test:
	nose2 -v --log-capture

.PHONY: build
build:
	python -m build --skip-dependency-check --no-isolation

.PHONY: build-zip
build-zip:
	@mkdir -p dist
	@name=$$(poetry version | awk '{print $$1}'); \
	ver=$$(poetry version -s); \
	echo "Building dist/$$name-$$ver.zip"; \
	cd $(CURDIR) && zip -r "dist/$$name-$$ver.zip" $$name -x '*__pycache__*' '*.pyc' >/dev/null; \
	echo "OK -> dist/$$name-$$ver.zip"

.PHONY: install
install:
	pip install dist/*.whl

.PHONY: uninstall
uninstall:
	pip uninstall admet_ai -y

.PHONY: qed
qed: clean build uninstall install
	pip list | grep "admet_ai"

.PHONY: compile
compile:
	python -m compileall -f ./admet_ai

.PHONY: clean
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
	rm -fr .ruff_cache/
	rm -f .coverage
	rm -f .noseids
	rm -f .profile
	rm -fr htmlcov

.PHONY: gcs-upload
gcs-upload: build build-zip
	@set -e; \
	[ -d dist ] && [ "$$(ls -A dist)" ] || { echo "No files in dist/"; exit 1; }; \
	gsutil cp dist/* "gs://$(BUCKET)/"