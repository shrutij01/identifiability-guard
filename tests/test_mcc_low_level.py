"""Focused tests for MCC low-level utilities and compatibility exports."""

import numpy as np
import pytest

from identifiability_guard.metrics.mcc import (
    copula_projection,
    largest_cancorr,
    make_diag,
    permutation_null_np,
    rank_array,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_legacy_helpers_remain_importable():
    """The module-to-package refactor must preserve the previous imports."""
    np.testing.assert_array_equal(
        make_diag(2, 2, 3),
        np.array([[2, 0, 0], [0, 2, 0]]),
    )
    np.testing.assert_array_equal(
        rank_array(np.array([30, 10, 20])),
        np.array([3, 1, 2]),
    )

    rng = np.random.default_rng(4)
    x = rng.normal(size=20)
    projected = copula_projection(x, k=5, rng=np.random.default_rng(5))
    assert projected.shape == (20, 6)

    matrix = rng.normal(size=(20, 3))
    assert largest_cancorr(matrix, matrix) == pytest.approx(1.0)


def test_legacy_permutation_null_does_not_require_crossfit_sample_size():
    """Legacy calibration must not construct unused K-fold splits."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 2))
    y = rng.normal(size=(6, 2))

    result = permutation_null_np(
        x,
        y,
        observed_score=0.2,
        metric="legacy",
        n_permutations=7,
        seed=7,
    )

    assert result.null_scores.shape == (7,)
    assert np.all(np.isfinite(result.null_scores))
    assert 0.0 < result.p_value_upper <= 1.0
    assert result.adjusted_score is not None


def test_crossfit_permutation_null_is_reproducible():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(60, 3))
    y = rng.normal(size=(60, 5))
    kwargs = {
        "observed_score": 0.1,
        "metric": "crossfit",
        "n_permutations": 7,
        "n_splits": 5,
        "seed": 9,
    }

    first = permutation_null_np(x, y, **kwargs)
    second = permutation_null_np(x, y, **kwargs)

    np.testing.assert_array_equal(first.null_scores, second.null_scores)
    assert first.p_value_upper == second.p_value_upper
    assert first.adjusted_score is None


@pytest.mark.parametrize("n_permutations", [0, -1])
def test_permutation_null_rejects_empty_null(n_permutations):
    values = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="at least 1"):
        permutation_null_np(
            values,
            values,
            observed_score=1.0,
            metric="legacy",
            n_permutations=n_permutations,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestRectangularTorchMatching:
    @pytest.mark.parametrize("shape", [(3, 3), (3, 6), (6, 3)])
    def test_sinkhorn_rectangular_plan(self, shape):
        from identifiability_guard.metrics.mcc import (
            sinkhorn_rectangular_plan_pt,
            sinkhorn_soft_mcc_pt,
        )

        generator = torch.Generator().manual_seed(10)
        affinity = torch.rand(shape, generator=generator)
        plan, scale, residual, iterations = sinkhorn_rectangular_plan_pt(
            affinity,
            max_iter=500,
        )

        d, m = shape
        assert tuple(plan.shape) == shape
        assert scale == pytest.approx(max(d, m) / min(d, m))
        assert float(plan.sum()) == pytest.approx(
            min(d, m) / max(d, m),
            abs=5e-4,
        )
        assert residual <= 2e-5
        assert 1 <= iterations <= 500

        score, metadata = sinkhorn_soft_mcc_pt(affinity, max_iter=500)
        assert 0.0 <= float(score) <= 1.0 + 1e-6
        assert metadata["marginal_residual"] <= 2e-5

    @pytest.mark.parametrize("shape", [(3, 6), (6, 3)])
    def test_padded_auction_path(self, shape):
        """Exercise padding directly; CPU dispatch otherwise uses SciPy."""
        from identifiability_guard.metrics.mcc._legacy import (
            _mcc_assignment_auction,
        )

        d, m = shape
        affinity = torch.zeros(shape, dtype=torch.float32)
        diagonal = torch.arange(min(d, m))
        affinity[diagonal, diagonal] = 1.0

        assert _mcc_assignment_auction(affinity, d, m) == pytest.approx(1.0)

    def test_sinkhorn_validates_parameters(self):
        from identifiability_guard.metrics.mcc import (
            sinkhorn_rectangular_plan_pt,
            sinkhorn_soft_mcc_pt,
        )

        affinity = torch.eye(2)
        with pytest.raises(ValueError, match="max_iter"):
            sinkhorn_rectangular_plan_pt(affinity, max_iter=0)
        with pytest.raises(ValueError, match="tol"):
            sinkhorn_rectangular_plan_pt(affinity, tol=0)
        with pytest.raises(ValueError, match="finite"):
            sinkhorn_rectangular_plan_pt(
                torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
            )
        with pytest.raises(ValueError, match="same shape"):
            sinkhorn_soft_mcc_pt(affinity, torch.eye(3))
