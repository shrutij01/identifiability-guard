# Installation

## Quick Install

Using `uv`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install
uv pip install -e .

# With dev tools
uv pip install -e ".[dev]"
```

## Standard Install

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

pip install -e .
pip install -e ".[dev]"    # With dev tools
```

## Verify

```bash
python -c "import identifiability_guard; print(identifiability_guard.__version__)"
pytest
```

## Dev Tools

Installing `.[dev]` gets you:
- pytest — testing
- black/isort — formatting
- flake8 — linting
- mypy — type checking
- pytest-cov — coverage

## Build Docs Locally

```bash
uv pip install -e ".[docs]"
mkdocs serve                # Auto-reload at http://127.0.0.1:8000
mkdocs build                # Static site in site/
mkdocs gh-deploy            # Deploy to GitHub Pages
```

Doc tools:
- mkdocs — static site generator
- mkdocs-material — theme
- mkdocstrings — auto API docs from docstrings

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20.0
- scipy ≥ 1.8.0
- scikit-learn ≥ 1.0.1
- matplotlib ≥ 3.5.0

## Troubleshooting

**Import errors after install**
```bash
which python  # Should show .venv/bin/python
```

**uv command not found**
```bash
export PATH="$HOME/.cargo/bin:$PATH"
# Then restart terminal
```

**Tests failing**
```bash
uv pip install -e .
pytest -v
```
