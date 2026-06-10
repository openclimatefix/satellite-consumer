.PHONY: init
init:
	@git config --local core.hooksPath .github/hooks
	@uv --version &> /dev/null || (echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" && exit 1)
	@uv sync

.PHONY: lint.dryrun
lint.dryrun:
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy .

.PHONY: lint
lint:
	@uv run ruff check --fix .
	@uv run ruff format .
	@uv run mypy .

.PHONY: test
test:
	@uv run python -m xmlrunner discover -s src/satellite_consumer -p "test_*.py" --output-file="unit-tests.xml" --outsuffix=""


