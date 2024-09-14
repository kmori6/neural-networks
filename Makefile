.PHONY: install test

install:
	poetry install

test:
	poetry run pytest --cov=neural_networks tests
