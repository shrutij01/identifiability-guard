"""Shared numerical constants and helpers for metric stability."""

import numpy as np
import warnings

EPS = 1e-12
SCALE_EPS_FACTOR = 1e-10


def safe_entropy_eps(values: np.ndarray) -> float:
    """Scale-aware epsilon for entropy smoothing."""
    return max(EPS, float(np.max(np.abs(values))) * SCALE_EPS_FACTOR)


def safe_divide(numerator, denominator, fallback=0.0):
    """Division with epsilon floor on denominator."""
    if abs(denominator) < EPS:
        return fallback
    return numerator / denominator


def clamp(value, lo=0.0, hi=1.0):
    """Clamp scalar to [lo, hi]."""
    return float(np.clip(value, lo, hi))


def sanitize_mi(mi_value: float) -> float:
    """Clamp MI to [0, inf), replacing NaN/negative with 0."""
    if not np.isfinite(mi_value) or mi_value < 0:
        return 0.0
    return float(mi_value)


def warn_nan(metric_name: str, context: str, count: int):
    """Consistent NaN warning across metrics."""
    if count > 0:
        warnings.warn(
            f"{metric_name}: {count} {context} produced NaN/non-finite values "
            f"and were excluded from the score."
        )
