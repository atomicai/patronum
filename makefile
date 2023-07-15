NAME := $(shell python setup.py --name)
UNAME := $(shell uname -s)

FLAKE_FLAGS=--in-place --remove-all-unused-imports --remove-unused-variable --recursive
# "" is for multi-lang strings (comments, logs), '' is for everything else.
# BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings

install:
	pip install -e '.[all]'

init:
	pip install pre-commit==3.3.3
	pip install isort==5.12.0
	pip install black==23.7.0
	pip install autoflake==2.2.0
	pre-commit clean
	pre-commit install
  	# To check whole pipeline.
	# pre-commit run --all-files


format:
	black external vkai test
	isort external vkai test
	autoflake ${FLAKE_FLAGS} external vkai test

test:
	pytest test ${PYTEST_FLAGS} --testmon --suppress-no-test-exit-code

test-all:
	pytest test ${PYTEST_FLAGS}

clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf downloads
	rm -rf wandb
	find . -name ".DS_Store" -print -delete
	rm -rf .cache
	pyclean .
