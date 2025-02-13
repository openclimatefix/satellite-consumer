# --- Dep builder image (can use python-3.12 if need gcc etc) --- #
FROM python:3.12-slim-bookworm AS build-deps

# Install build requirements into build image
# * UV for python packaging
# * git for setuptools-git-versioning
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y git

# Add only files required for dependencies
# * pyproject.toml: Project configuration
WORKDIR /opt/app
COPY pyproject.toml /opt/app/pyproject.toml

# Make UV behave in a container-orientated way
# * Compile bytecode to reduce startup time
# * Disable cache to reduce image size
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# Install dependecies
# * --no-dev: Do not install development dependencies
# * --no-install-project: Only install dependencies
# * --no-editable: Copy the source code into site-packages
RUN mkdir src && \
    uv sync --no-dev --no-install-project --no-editable

# Remove tests (Pandas ship loads, for instance)
# * Remove this line if causing problems
RUN rm -rf /opt/app/.venv/lib/python3.12/site-packages/**/tests

# --- App builder image --- #
FROM build-deps AS build-app

# Install the project
# * .git: Required for setuptools-git-versioning
COPY src /opt/app/src
COPY .git /opt/app/.git
RUN uv sync --no-dev --no-editable

# --- Runtime image (use distroless if feasible for 100MB saving) --- #
FROM python:3.12-slim-bookworm
# FROM al3xos/python-distroless:3.12-debian12

WORKDIR /opt/app
# Copy just the virtual environment into a runtime image
COPY --from=build-app --chown=app:app /opt/app/.venv /opt/app/.venv

ENTRYPOINT ["/opt/app/.venv/bin/ocf-template-cli"]

