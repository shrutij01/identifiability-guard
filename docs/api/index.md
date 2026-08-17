# API Reference

The package is organized into four public modules. Each page below is generated
from the implementation and its docstrings.

## Data-generating processes

[`identifiability_guard.dgp`](dgp.md) contains `BaseDGP` and the D1–D4 factor
generators: `D1Independent`, `D2Correlated`, `D3SingleRedundant`, and
`D4MultiRedundant`.

## Encoders

[`identifiability_guard.encoders`](encoders.md) contains `BaseEncoder` and the
E1–E10 controlled transformations, including exact, undercomplete,
overcomplete, entangled, and null encoders.

## Metrics

[`identifiability_guard.metrics`](metrics.md) contains the common `BaseMetric`
and `MetricResult` interface, MCC, DCI, R², MIG, T-MEX, InfoMEC, and
`MetricRegistry`.

## Evaluation utilities

[`identifiability_guard.evaluation`](evaluation.md) contains timing helpers,
multi-seed aggregation, sensitivity sweeps, and shared experiment helpers.
