# Identifiability Guard

A modular framework for evaluating identifiability metrics in representation learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Identifiability Guard provides a systematic approach to studying how well learned representations recover ground-truth latent factors. The framework consists of three core components:

1. **Data Generating Processes (DGPs)** - Generate ground-truth latent factors with controlled statistical properties
2. **Encoder Mixings** - Transform latents to representations in systematic ways
3. **Identifiability Metrics** - Quantify recovery quality

## Quick Start

### Installation

```bash
# Using uv (recommended - 10-100x faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

See [Installation Guide](docs/installation.md) for detailed instructions.

### Basic Usage

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import MCC, DCI

# Generate ground-truth factors
dgp = D1Independent(d=5, seed=42)
Z = dgp.sample(1000)

# Apply encoder mixing
encoder = E1ElementwiseLinear(d=5, seed=42)
Z_hat = encoder.encode(Z)

# Evaluate identifiability
mcc = MCC()
dci = DCI()

print(f"MCC: {mcc.compute(Z, Z_hat).primary_score:.3f}")
print(f"DCI: {dci.compute(Z, Z_hat).subscores['disentanglement']:.3f}")
```

## Documentation

📚 **[Full Documentation](docs/index.md)**

- [Installation Guide](docs/installation.md)
- [Data Generating Processes](docs/dgp.md)
- [Encoder Mixings](docs/encoders.md)
- [Identifiability Metrics](docs/metrics.md)
- [Examples & Tutorials](docs/examples.md)
- [API Reference](docs/api/index.md)
- [Contributing Guide](docs/contributing.md)

## Components

### Data Generating Processes (DGPs)

| DGP | Description |
|-----|-------------|
| D1 | Independent, non-redundant factors |
| D2 | Correlated, non-redundant factors |
| D3 | Single-factor redundant |
| D4 | Multi-factor redundant |

[See full DGP documentation →](docs/dgp.md)

### Encoder Mixings

| Encoder | Description | Dimensionality |
|---------|-------------|----------------|
| E1 | Elementwise Linear | m = d |
| E2 | Elementwise Nonlinear | m = d |
| E3 | Linearly Entangled | m = d |
| E4 | Undercomplete Linear | m < d |
| E5 | Overcomplete Linear | m > d |
| E6 | Overcomplete Multicodes | m > d |
| E7 | Overcomplete Entangled | m > d |
| E8 | Overcomplete Disjoint | m > d |
| E9/E10 | Random Baselines | m = d |

[See full encoder documentation →](docs/encoders.md)

### Identifiability Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| MCC | Mean Correlation Coefficient | [0, 1] |
| DCI | Disentanglement, Completeness, Informativeness | [0, 1] |
| R² | Coefficient of Determination | [0, 1] |
| MIG | Mutual Information Gap | [0, ∞) |
| T-MEX | Testing for Measurement Exchangeability | [0, 1] |
| InfoMEC | Modularity, Explicitness, Compactness | [0, 1] |

[See full metrics documentation →](docs/metrics.md)

## Examples

Run comprehensive evaluations:

```bash
# Generate DGP × Encoder heatmap
python examples/evaluate_all_combinations_combined.py

# Sample size sensitivity analysis
python examples/evaluate_sensitivity.py \
    --sweep-samples 500,1000,5000,10000 \
    --dgp D1 --encoder E1 --n-seeds 10
```

[See more examples →](docs/examples.md)

## Development

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/ && isort src/ tests/

# Type check
mypy src/
```

[See contributing guide →](docs/contributing.md)

## Project Structure

```
identifiability-guard/
├── src/identifiability_guard/   # Source code
│   ├── dgp/                      # Data generating processes
│   ├── encoders/                 # Encoder mixings
│   ├── metrics/                  # Identifiability metrics
│   └── evaluation/               # Evaluation utilities
├── tests/                        # Unit tests
├── examples/                     # Example scripts
├── docs/                         # Documentation
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{identifiability_guard,
  title = {Identifiability Guard: A Framework for Evaluating Identifiability Metrics},
  author = {Identifiability Guard Team},
  year = {2024},
  url = {https://github.com/yourusername/identifiability-guard}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This framework implements and extends identifiability metrics from the representation learning literature. See individual metric documentation for references.
