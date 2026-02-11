# Documentation Overview

Welcome to the Identifiability Guard documentation!

## Getting Started

1. **[Installation](installation.md)** - Setup instructions for uv and pip
2. **[Quick Start](index.md)** - Basic usage example
3. **[Examples](examples.md)** - Practical usage patterns
4. **[Experiments](experiments.md)** - Running experiment scripts and sweeps

## Core Components

### [Data Generating Processes (DGPs)](dgp.md)
Generate ground-truth latent factors with controlled statistical properties:
- D1: Independent factors
- D2: Correlated factors
- D3: Single-factor redundant
- D4: Multi-factor redundant

### [Encoder Mixings](encoders.md)
Transform latents to representations in systematic ways:
- E1-E3: Exact dimensionality (m = d)
- E4: Undercomplete (m < d)
- E5-E8: Overcomplete (m > d)
- E9-E10: Random baselines

### [Identifiability Metrics](metrics.md)
Quantify how well representations recover ground-truth factors:
- MCC: Mean Correlation Coefficient
- DCI: Disentanglement, Completeness, Informativeness
- R²: Coefficient of Determination
- MIG: Mutual Information Gap
- T-MEX: Testing for Measurement Exchangeability
- InfoMEC: Modularity, Explicitness, Compactness

## Reference

- **[API Reference](api/index.md)** - Complete API documentation
- **[Contributing](contributing.md)** - Development guidelines

## Navigation

```
docs/
├── index.md           # Documentation home
├── dgp.md             # Data Generating Processes
├── encoders.md        # Encoder Mixings
├── metrics.md         # Identifiability Metrics
├── examples.md        # Usage examples
├── contributing.md    # Development guide
└── api/               # API reference
    └── index.md
```

## Quick Links

- [GitHub Repository](https://github.com/yourusername/identifiability-guard)
- [Issue Tracker](https://github.com/yourusername/identifiability-guard/issues)
- [Discussions](https://github.com/yourusername/identifiability-guard/discussions)
