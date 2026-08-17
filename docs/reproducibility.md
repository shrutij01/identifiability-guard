# Reproducibility

Reproducibility has two separate sources of randomness in this project:

1. sampling factors and constructing encoder transformations; and
2. metrics that use random splits, estimators, projections, or tests.

Seed both layers explicitly. Do not rely on NumPy's global random state.

## Reproducible end-to-end evaluation

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.metrics import DCI, InfoMEC, MCC, MIG, R2, TMEX

seed = 42

dgp = D1Independent(d=5, seed=seed)
Z = dgp.sample(1000)

encoder = E1ElementwiseLinear(d=5, seed=seed)
Z_hat = encoder.encode(Z)

metrics = {
    "mcc_pearson": MCC(method="pearson"),
    "mcc_rdc": MCC(method="rdc", seed=seed),
    "dci": DCI(random_state=seed),
    "r2": R2(),
    "mig": MIG(num_bins=20),
    "tmex": TMEX(seed=seed),
    "infomec": InfoMEC(random_state=seed),
}

scores = {
    name: metric.compute(Z, Z_hat).primary_score
    for name, metric in metrics.items()
}
```

Constructing the same DGP, encoder, and seeded metrics again produces the same
results on the same software stack.

## Which metrics need a seed?

| Metric | Seed argument | Why |
|---|---|---|
| MCC-Pearson / MCC-Spearman | `seed` when `crossfit=True` | Controls cross-validation folds |
| MCC-RDC | `seed` | Controls RDC random projections |
| DCI | `random_state` | Controls the train/test split and gradient-boosting estimators |
| InfoMEC / InfoM / InfoE / InfoC | `random_state` | Controls mutual-information estimation and predictive models |
| T-MEX | `seed` | Controls repeated conditional-independence tests |
| MIG | none | Histogram-based implementation is deterministic |
| R² | none | Least-squares implementation is deterministic |

For seeded metrics, leaving the seed unset is supported but does not promise
repeatability.

## Repeating experiments across seeds

Use an explicit seed list and return scalar values from the evaluation
function:

```python
from identifiability_guard.dgp import D1Independent
from identifiability_guard.encoders import E1ElementwiseLinear
from identifiability_guard.evaluation import run_with_seeds
from identifiability_guard.metrics import DCI, MCC


def evaluate(seed):
    Z = D1Independent(d=5, seed=seed).sample(1000)
    Z_hat = E1ElementwiseLinear(d=5, seed=seed).encode(Z)
    return {
        "mcc": MCC().compute(Z, Z_hat).primary_score,
        "dci": DCI(random_state=seed).compute(Z, Z_hat).primary_score,
    }


raw_results = run_with_seeds(
    evaluate,
    seeds=[0, 1, 2, 3, 4],
    n_jobs=1,
    verbose=False,
)
```

Keep the seed list in the experiment configuration alongside all other sweep
parameters. Sorting input file paths before iteration also prevents accidental
order dependence.

## Resetting generators

DGPs and encoders keep an internal NumPy generator. To replay their sequence,
either create a new instance with the same seed or call `reset_rng()`:

```python
dgp = D1Independent(d=5, seed=42)
first = dgp.sample(100)

dgp.reset_rng()
replayed = dgp.sample(100)
```

## Environment and numerical precision

Fixed seeds control algorithmic randomness, but floating-point results can
still differ slightly across NumPy, SciPy, scikit-learn, BLAS, operating-system,
and CPU versions. For archival runs:

```bash
python --version
python -m pip freeze > environment.txt
```

Record the Git commit, the command, the seed list, and the generated config
file. Compare cross-platform floating-point outputs with a tolerance rather
than requiring bitwise equality.

## Verification

The test suite includes seeded reproducibility checks:

```bash
pytest tests/test_metrics.py::TestReproducibility
```

For a complete release check, run `pytest` from an editable development
installation.
