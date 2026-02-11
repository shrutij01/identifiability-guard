# Identifiability Guard Documentation

A modular framework for evaluating identifiability metrics in representation learning.

## Overview

Identifiability Guard provides a systematic approach to studying how well learned representations recover ground-truth latent factors. The framework is built around three core components:

1. **[Data Generating Processes (DGPs)](dgp.md)** - Generate ground-truth latent factors with controlled statistical properties
2. **[Encoder Mixings](encoders.md)** - Transform latents to representations in systematic ways
3. **[Identifiability Metrics](metrics.md)** - Quantify recovery quality

## Quick Links

- [Installation](installation.md)
- [API Reference](api/index.md)
- [Examples](examples.md)
- [Contributing](contributing.md)

## Quick Example

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

mcc_score = mcc.compute(Z, Z_hat)
dci_scores = dci.compute(Z, Z_hat)

print(f"MCC: {mcc_score.primary_score:.3f}")
print(f"DCI Disentanglement: {dci_scores.subscores['disentanglement']:.3f}")
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

MIT License - see [LICENSE](../LICENSE) for details.
