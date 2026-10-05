# Development Guide

## Setup

Cube Wrangler uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/network-wrangler/cube_wrangler
cd cube_wrangler
uv sync
```

This installs all dev dependencies (pytest, ruff, mypy, pre-commit, pytest-benchmark).

### Pre-commit hooks

```bash
uv run pre-commit install
```

## Running Tests

```bash
# All tests (excluding slow benchmarks)
uv run pytest tests/ -m "not benchmark"

# Single test file
uv run pytest tests/test_roadway.py -v

# Single test
uv run pytest tests/test_roadway.py::test_parameter_read -v
```

## Benchmarks

Performance benchmarks are marked with `@pytest.mark.benchmark` and excluded from the default CI run.

```bash
# Run benchmarks
uv run pytest tests/test_benchmark.py -m benchmark

# Save results for branch comparison
uv run pytest tests/test_benchmark.py -m benchmark --benchmark-save=my_branch

# Compare saved results
pytest-benchmark compare branch_a branch_b
```

Benchmark test data (synthetic Cube log files) lives in `tests/data/`. If the files are missing, the fixtures regenerate them automatically from the stpaul example network.

To regenerate manually:

```bash
uv run python benchmarks/generate_test_logfile.py --sizes 10,50,100,500,1000
```

## Linting and Formatting

```bash
# Check
uv run ruff check cube_wrangler
uv run ruff format --check cube_wrangler

# Auto-fix
uv run ruff check --fix cube_wrangler
uv run ruff format cube_wrangler
```

## Docs

```bash
# Serve locally with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Install doc dependencies first:

```bash
uv pip install -e .[docs]
```

## Versioning and Releases

Cube Wrangler uses a static version in `cube_wrangler/__init__.py` and `pyproject.toml`. Both must match the git release tag.

To release a new version:

1. Update `__version__` in `cube_wrangler/__init__.py`
2. Update `version` in `pyproject.toml`
3. Commit and push: `git commit -m "chore: bump version to X.Y.Z"`
4. Create a GitHub Release with tag `vX.Y.Z`

The `prepare-release` workflow validates the version, publishes to TestPyPI, and tests installation. The `publish` workflow publishes to PyPI and deploys docs when the release is published.

### GitHub Environments

The release workflows use [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/) — no stored secrets needed. Set up two GitHub Environments:

- **testpypi**: linked to `https://test.pypi.org/p/cube-wrangler`
- **pypi**: linked to `https://pypi.org/p/cube-wrangler`
