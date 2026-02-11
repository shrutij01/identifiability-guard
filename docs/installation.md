# Installation Guide

## Quick Start (Recommended: uv)

[uv](https://github.com/astral-sh/uv) is a blazing-fast Python package installer (10-100x faster than pip).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install package in editable mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

## Standard Installation (pip)

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install package in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Test import
python -c "import identifiability_guard; print(identifiability_guard.__version__)"

# Run tests
pytest
```

## Development Setup

```bash
# Install with all development tools
uv pip install -e ".[dev]"  # or pip install -e ".[dev]"

# Development tools included:
# - pytest: Testing framework
# - pytest-cov: Coverage reporting
# - black: Code formatter
# - isort: Import sorter
# - flake8: Linting
# - mypy: Type checking
```

## Documentation Setup

Build and serve the documentation website locally:

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"  # or pip install -e ".[docs]"

# Serve docs locally with auto-reload
mkdocs serve
# Open http://127.0.0.1:8000 in your browser

# Build static documentation site
mkdocs build
# Output will be in site/ directory

# Deploy to GitHub Pages (optional)
mkdocs gh-deploy
```

Documentation tools included:
- **mkdocs**: Static site generator
- **mkdocs-material**: Modern, responsive theme
- **mkdocstrings**: Auto-generates API docs from docstrings

## From PyPI (Coming Soon)

```bash
uv pip install identifiability-guard  # or pip install identifiability-guard
```

## System Requirements

- Python ≥ 3.8
- Operating System: macOS, Linux, or Windows

## Dependencies

Core dependencies (installed automatically):
- numpy ≥ 1.20.0
- scipy ≥ 1.8.0
- scikit-learn ≥ 1.0.1
- matplotlib ≥ 3.5.0

## Troubleshooting

**Issue**: Import errors after installation
```bash
# Solution: Ensure you're in the activated virtual environment
which python  # Should show .venv/bin/python
```

**Issue**: uv command not found
```bash
# Solution: Restart terminal or add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**Issue**: Tests failing
```bash
# Solution: Reinstall in editable mode
uv pip install -e .
pytest -v
```
