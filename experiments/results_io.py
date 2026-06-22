"""Efficient save/load for experiment results.

Numeric data is stored as compressed NPZ (fast I/O, compact on disk).
Non-numeric metadata (axis labels, config) goes into a JSON sidecar.

Usage
-----
    from results_io import save_results, load_results

    # After computing:
    save_results("exp01", {"means": means, "ci_lo": ci_lo, ...},
                 config={"dgp_names": [...], ...})

    # To reload later (e.g. for re-plotting):
    data, config = load_results("exp01")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "runs"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten a nested dict into ``{key1__key2: np.array}``.

    Leaf values must be numeric (scalars, lists, or numpy arrays).
    Non-numeric or ``None`` values are silently skipped.
    """
    items: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        new_key = f"{prefix}__{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        elif isinstance(v, (list, np.ndarray, int, float, np.integer, np.floating)):
            items[new_key] = np.asarray(v)
    return items


def _unflatten(flat_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Reconstruct a nested dict from flattened ``__``-separated keys.

    Zero-dimensional arrays are converted to Python scalars for
    cleaner downstream use (e.g. ``scores[dgp][enc][met]`` returns a
    plain ``float`` instead of a 0-d array).
    """
    result: Dict[str, Any] = {}
    for key, val in flat_dict.items():
        if isinstance(val, np.ndarray) and val.ndim == 0:
            val = val.item()
        parts = key.split("__")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return result


def _json_default(obj):
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_results(
    exp_name: str,
    data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save experiment results to compressed NPZ + JSON config.

    Parameters
    ----------
    exp_name : str
        Experiment identifier (e.g. ``"exp01"``).  Results are stored
        under ``runs/<exp_name>/``.
    data : dict
        Arbitrarily nested dict whose leaf values are numeric
        (lists, arrays, scalars).  Nested keys are joined with ``__``
        in the NPZ archive.
    config : dict, optional
        Non-numeric metadata (axis labels, parameter lists, etc.)
        stored as a JSON sidecar.

    Returns
    -------
    Path
        Directory where results were written.
    """
    out_dir = RESULTS_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = _flatten(data)
    np.savez_compressed(out_dir / "results.npz", **flat)

    if config is not None:
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=_json_default)

    print(f"Results saved -> {out_dir}")
    return out_dir


def load_results(
    exp_name: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load previously saved experiment results.

    Returns
    -------
    data : dict
        Nested dict matching the structure originally passed to
        :func:`save_results`.  Array leaves are numpy arrays; scalar
        leaves are plain Python floats/ints.
    config : dict or None
        The JSON config sidecar, if it exists.
    """
    out_dir = RESULTS_DIR / exp_name
    npz_path = out_dir / "results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No saved results for {exp_name!r} at {npz_path}"
        )

    with np.load(npz_path, allow_pickle=False) as npz:
        flat = {k: npz[k] for k in npz.files}
    data = _unflatten(flat)

    config = None
    config_path = out_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return data, config


def results_exist(exp_name: str) -> bool:
    """Check whether saved results exist for an experiment."""
    return (RESULTS_DIR / exp_name / "results.npz").exists()
