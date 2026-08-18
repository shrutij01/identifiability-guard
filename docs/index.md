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
- [Experiments](experiments.md)
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
@InProceedings{pmlr-v337-joshi26a,
  title = {Who Guards the Guardians? {The} Challenges of Evaluating Identifiability of Learned Representations},
  author = {Joshi, Shruti and Saulus, Th\'{e}o and Brendel, Wieland and Brouillard, Philippe and Sridhar, Dhanya and Reizinger, Patrik},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages = {2618--2660},
  year = {2026},
  editor = {Perković, Emilija and Malinsky, Daniel},
  volume = {337},
  series = {Proceedings of Machine Learning Research},
  month = {17--21 Aug},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v337/main/assets/joshi26a/joshi26a.pdf},
  url = {https://proceedings.mlr.press/v337/joshi26a.html},
  abstract = {Identifiability in representation learning is commonly evaluated using standard metrics (e.g., *MCC, $R^2$, DCI*) on synthetic benchmarks with known ground-truth factors. These metrics are assumed to reflect recovery up to the equivalence class guaranteed by identifiability theory. We show that this assumption holds only under specific structural conditions: each metric implicitly encodes assumptions about both the data-generating process ({DGP}) and the encoder. When these assumptions are violated, metrics become misspecified and can produce systematic false positives and false negatives. Such failures occur both within classical identifiability regimes and in post-hoc settings where identifiability is most needed. We introduce a taxonomy separating {DGP} assumptions from encoder geometry, use it to characterise the validity domains of existing metrics, and release an evaluation suite for reproducible stress testing and comparison.}
}
```

See the [paper](https://proceedings.mlr.press/v337/joshi26a.html) and the repository's
[`CITATION.cff`](https://github.com/shrutij01/identifiability-guard/blob/main/CITATION.cff)
for the preferred paper citation.

## License

MIT License - see
[LICENSE](https://github.com/shrutij01/identifiability-guard/blob/main/LICENSE)
for details.
