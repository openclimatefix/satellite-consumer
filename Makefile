.PHONY: init
init:
	@git config --local core.hooksPath .github/hooks
	@uv --version &> /dev/null || (echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" && exit 1)
	@uv sync

.PHONY: lint.dryrun
lint.check:
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy .

.PHONY: lint
format:
	@uv run ruff check --fix .
	@uv run ruff format .

.PHONY: test
test:
	@uv run python -m xmlrunner discover -s src/satellite_consumer -p "test_*.py" -o unit-tests.xml


