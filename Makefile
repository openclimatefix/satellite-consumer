.PHONY: test lint typecheck

test:
	uv run python -m unittest discover -s src/satellite_consumer -p "test_*.py"

test-cov:
	uv run python -m xmlrunner discover -s src/satellite_consumer -p "test_*.py"

lint:
	uv run ruff check --fix .

typecheck:
	uv run mypy .

