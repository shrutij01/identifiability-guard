"""
Identifiability Metrics module.

Provides metrics for evaluating identifiability:
- DCI: Disentanglement, Completeness, Informativeness
- MCC: Mean Correlation Coefficient (Pearson, Spearman, RDC)
- R²: Coefficient of Determination
- MIG: Mutual Information Gap
- T-MEX: Testing for Measurement Exchangeability
- InfoMEC: Information-theoretic Modularity, Explicitness, Compactness (InfoM, InfoE, InfoC)

All metrics inherit from BaseMetric and return MetricResult objects.
Use MetricRegistry for unified API across all metrics.

Example:
    >>> from identifiability_guard.metrics import MetricRegistry
    >>> registry = MetricRegistry()
    >>> registry.register_defaults()
    >>>
    >>> # Use a single metric
    >>> metric = registry.create("dci")
    >>> result = metric.compute(Z, Z_hat)
    >>> print(result.primary_score)
    >>>
    >>> # Or compute all at once
    >>> results = registry.compute_all(Z, Z_hat)
"""

from .base import BaseMetric, MetricResult
from .dci import DCIMetric
from .mcc import MCCMetric
from .r2 import R2Metric
from .mig import MIGMetric
from .tmex import TMEXMetric
from .infomec import InfoMECMetric, InfoMMetric, InfoEMetric, InfoCMetric
from .registry import MetricRegistry

# Aliases for convenience
DCI = DCIMetric
MCC = MCCMetric
R2 = R2Metric
MIG = MIGMetric
TMEX = TMEXMetric
InfoMEC = InfoMECMetric
InfoM = InfoMMetric
InfoE = InfoEMetric
InfoC = InfoCMetric

__all__ = [
    # Core classes
    "BaseMetric",
    "MetricResult",
    # Metrics
    "DCIMetric",
    "MCCMetric",
    "R2Metric",
    "MIGMetric",
    "TMEXMetric",
    "InfoMECMetric",
    "InfoMMetric",
    "InfoEMetric",
    "InfoCMetric",
    # Aliases
    "DCI",
    "MCC",
    "R2",
    "MIG",
    "TMEX",
    "InfoMEC",
    "InfoM",
    "InfoE",
    "InfoC",
    # Registry
    "MetricRegistry",
]
