# Contributing Guide

## Development Setup

```bash
# Clone repository
git clone https://github.com/shrutij01/identifiability-guard.git
cd identifiability-guard

# Install with development dependencies
uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"

# Verify installation
pytest
```

## Building Documentation

Documentation is auto-generated from docstrings using MkDocs.

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve docs locally (auto-reload on changes)
mkdocs serve
# Then open http://127.0.0.1:8000

# Build static site
mkdocs build
# Output in site/ directory

# Deploy to GitHub Pages
mkdocs gh-deploy
```

The documentation uses:
- **MkDocs Material** - Modern, responsive theme
- **mkdocstrings** - Auto-generates API docs from docstrings
- **Google-style docstrings** - Standard format for all code

## Code Quality Tools

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=identifiability_guard --cov-report=html

# Specific test file
pytest tests/test_metrics.py

# Verbose output
pytest -v
```

## Adding New Components

### Adding a New DGP

1. Create file in `src/identifiability_guard/dgp/`
2. Inherit from `BaseDGP`
3. Implement `sample()` method

```python
from .base import BaseDGP
from typing import Optional
import numpy as np

class D5MyNewDGP(BaseDGP):
    """Description of the DGP."""

    def __init__(
        self,
        d: int,
        my_param: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(d=d, seed=seed)
        self.my_param = my_param

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples.

        Args:
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples, d)
        """
        raise NotImplementedError
```

4. Add to `src/identifiability_guard/dgp/__init__.py`:

```python
from .d5_my_new_dgp import D5MyNewDGP

__all__ = [
    ...,
    "D5MyNewDGP",
]
```

5. Add tests in `tests/test_dgp.py`

### Adding a New Encoder

1. Create file in `src/identifiability_guard/encoders/`
2. Inherit from `BaseEncoder`
3. Implement `encode()` method

```python
from .base import BaseEncoder
from typing import Optional
import numpy as np

class E11MyNewEncoder(BaseEncoder):
    """Description of the encoder."""

    def __init__(
        self,
        d: int,
        m: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(d=d, m=m, seed=seed)

    def _initialize_parameters(self) -> None:
        """Initialize parameters used by the transformation."""
        self._initialized = True

    def encode(self, Z: np.ndarray) -> np.ndarray:
        """
        Encode latent factors.

        Args:
            Z: Ground-truth factors of shape (n_samples, d)

        Returns:
            Encoded representations of shape (n_samples, m)
        """
        raise NotImplementedError
```

4. Add to `src/identifiability_guard/encoders/__init__.py`
5. Add tests in `tests/test_encoders.py`

### Adding a New Metric

1. Create file in `src/identifiability_guard/metrics/`
2. Inherit from `BaseMetric`
3. Implement `_compute_impl()` returning `MetricResult`; `BaseMetric.compute()`
   handles input and output validation

```python
from .base import BaseMetric, MetricResult
import numpy as np

class MyNewMetric(BaseMetric):
    """Description of the metric."""

    def __init__(self, param: float = 1.0):
        self.param = param

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        """
        Compute metric.

        Args:
            Z: Ground-truth factors of shape (n_samples, d)
            Z_hat: Learned representations of shape (n_samples, m)

        Returns:
            MetricResult with primary_score, subscores, metadata
        """
        raise NotImplementedError
```

4. Add to `src/identifiability_guard/metrics/__init__.py`
5. Register in `MetricRegistry` if desired
6. Add tests in `tests/test_metrics.py`

## Testing Guidelines

### Unit Test Structure

```python
import pytest
import numpy as np
from identifiability_guard.dgp import D1Independent

class TestD1Independent:
    def test_sample_shape(self):
        dgp = D1Independent(d=5, seed=42)
        Z = dgp.sample(100)
        assert Z.shape == (100, 5)

    def test_independence(self):
        dgp = D1Independent(d=5, seed=42)
        Z = dgp.sample(10000)
        corr = np.corrcoef(Z.T)
        off_diag = corr - np.eye(5)
        assert np.abs(off_diag).max() < 0.1

    def test_reproducibility(self):
        dgp1 = D1Independent(d=5, seed=42)
        dgp2 = D1Independent(d=5, seed=42)
        Z1 = dgp1.sample(100)
        Z2 = dgp2.sample(100)
        np.testing.assert_array_equal(Z1, Z2)
```

### Fixture Usage

```python
@pytest.fixture
def sample_data():
    dgp = D1Independent(d=5, seed=42)
    return dgp.sample(1000)

def test_metric_with_fixture(sample_data):
    Z = sample_data
    # Test using Z
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def compute_score(Z: np.ndarray, Z_hat: np.ndarray, normalize: bool = True) -> float:
    """
    Compute alignment score between representations.

    Args:
        Z: Ground-truth factors of shape (n_samples, d)
        Z_hat: Learned representations of shape (n_samples, m)
        normalize: Whether to normalize scores to [0, 1]

    Returns:
        Alignment score (higher is better)

    Raises:
        ValueError: If Z and Z_hat have different number of samples

    Example:
        >>> Z = np.random.randn(100, 5)
        >>> Z_hat = np.random.randn(100, 5)
        >>> score = compute_score(Z, Z_hat)
        >>> assert 0 <= score <= 1
    """
    pass
```

### Type Hints

Use type hints consistently:

```python
from typing import Optional, Dict, Tuple
import numpy as np

def process_data(
    Z: np.ndarray,
    normalize: bool = True,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process data and return results."""
    pass
```

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-new-feature`
3. **Make changes** following code quality guidelines
4. **Add tests** for new functionality
5. **Update documentation** in `docs/` if needed
6. **Run tests**: `pytest`
7. **Format code**: `black src/ tests/ && isort src/ tests/`
8. **Commit** with clear messages: `git commit -m "Add feature: description"`
9. **Push** to your fork: `git push origin feature/my-new-feature`
10. **Create Pull Request** on GitHub

### Commit Message Format

```
Add feature: Brief description (50 chars max)

More detailed explanation if needed. Wrap at 72 characters.
Include motivation for the change and contrast with previous
behavior.

- Bullet points are okay
- Use present tense ("Add" not "Added")
```

## Code Style

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `D1Independent`, `MCCMetric`)
- **Functions/methods**: `snake_case` (e.g., `compute_score`, `sample`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SEED`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_helper`)

### Import Organization

```python
# Standard library
import os
from typing import Optional

# Third-party
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso

# Local
from .base import BaseMetric
from ..dgp import D1Independent
```

### Function Length

Keep functions focused and concise:
- Aim for < 50 lines per function
- Extract complex logic into helper functions
- Use early returns to reduce nesting

## Performance Considerations

### Vectorization

Prefer NumPy vectorized operations:

```python
# Bad
result = np.array([x**2 for x in data])

# Good
result = data ** 2
```

### Memory Efficiency

```python
# Bad - creates intermediate copy
Z_centered = Z - Z.mean(axis=0)
Z_normalized = Z_centered / Z_centered.std(axis=0)

# Good - in-place operations
Z_centered = Z - Z.mean(axis=0)
Z_centered /= Z_centered.std(axis=0)
```

### Profiling

Use profiling tools to identify bottlenecks:

```python
from identifiability_guard.evaluation import profile_block

with profile_block("Expensive operation") as profile:
    result = expensive_computation()

print(f"Time: {profile['elapsed']:.2f}s")
print(f"Memory: {profile['peak_mb']:.2f} MB")
```

## Issue Reporting

When reporting bugs, include:
1. Python version and OS
2. Package version: `python -c "import identifiability_guard; print(identifiability_guard.__version__)"`
3. Minimal reproducible example
4. Expected vs actual behavior
5. Full error traceback

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
