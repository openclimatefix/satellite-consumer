.PHONY: test
test:
	uv run python -m unittest discover -s src/satellite_consumer -p "test_*.py"

.PHONY: test-cov
test-cov:
	uv run python -m xmlrunner discover -s src/satellite_consumer -p "test_*.py"

.PHONY: lint
lint:
	uv run ruff check --fix .

.PHONY: typecheck
typecheck:
	uv run mypy .

