"""
Evaluation utilities module.

Provides tools for:
- Timing and memory profiling (timing.py)
- Multi-seed evaluations with statistics (multi_seed.py)
- Sensitivity analysis and parameter sweeps (sensitivity.py)
- Shared evaluation helpers (helpers.py)
"""

from .timing import (
    time_block,
    memory_profiler,
    profile_block,
    timed,
    profiled,
    Timer,
)

from .multi_seed import (
    run_with_seeds,
    compute_statistics,
    aggregate_results,
    run_multi_seed_evaluation,
    format_result_with_ci,
    format_result_with_std,
)

from .sensitivity import (
    parameter_sweep,
    sensitivity_analysis_1d,
    compute_sensitivity_statistics,
    save_sensitivity_results,
    load_sensitivity_results,
)

from .helpers import (
    DGP_CLASSES,
    ENCODER_CLASSES,
    ALL_METRICS,
    METRIC_DISPLAY_NAMES,
    MAIN_METRICS,
    APX_METRICS,
    DEFAULT_METRICS,
    get_dgp_class,
    get_encoder_class,
    extract_metric_scores,
    validate_array,
    evaluate_combination,
    create_dgp_with_params,
    create_encoder_with_params,
)

__all__ = [
    # Timing utilities
    "time_block",
    "memory_profiler",
    "profile_block",
    "timed",
    "profiled",
    "Timer",
    # Multi-seed utilities
    "run_with_seeds",
    "compute_statistics",
    "aggregate_results",
    "run_multi_seed_evaluation",
    "format_result_with_ci",
    "format_result_with_std",
    # Sensitivity analysis utilities
    "parameter_sweep",
    "sensitivity_analysis_1d",
    "compute_sensitivity_statistics",
    "save_sensitivity_results",
    "load_sensitivity_results",
    # Shared helpers
    "DGP_CLASSES",
    "ENCODER_CLASSES",
    "ALL_METRICS",
    "METRIC_DISPLAY_NAMES",
    "MAIN_METRICS",
    "APX_METRICS",
    "DEFAULT_METRICS",
    "get_dgp_class",
    "get_encoder_class",
    "extract_metric_scores",
    "validate_array",
    "evaluate_combination",
    "create_dgp_with_params",
    "create_encoder_with_params",
]
