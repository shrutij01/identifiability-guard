"""Edge-case tests for numerical stability fixes."""

import numpy as np
import pytest
import warnings

from identifiability_guard.metrics import MIG, MCC, TMEX, InfoM, InfoE, InfoC, InfoMECMetric
from identifiability_guard.metrics.r2 import R2Metric
from identifiability_guard.metrics.dci import (
    disentanglement,
    completeness,
    disentanglement_per_code,
    completeness_per_factor,
)
from identifiability_guard.metrics.mig import _compute_mig, histogram_discretize, discrete_mutual_info
from identifiability_guard.metrics.mcc import corrcoef_pt, cov_pt


class TestMIGSingleCode:
    """MIG must not crash when num_codes == 1."""

    def test_single_code(self):
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z[:, :1]  # only 1 code
        mig = MIG()
        result = mig.compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1

    def test_compute_mig_single_code_direct(self):
        codes = np.random.randn(1, 500)  # 1 code, 500 samples
        factors = np.random.randn(3, 500)
        d_codes = histogram_discretize(codes)
        d_factors = histogram_discretize(factors)
        score, info = _compute_mig(d_codes, d_factors)
        assert np.isfinite(score)


class TestMCCConstantColumn:
    """MCC must handle constant columns without crashing."""

    def test_constant_column_numpy(self):
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = np.column_stack([Z[:, 0], np.ones(200), Z[:, 2]])
        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1


class TestMCCPyTorch:
    """PyTorch MCC guards for zero stddev and single sample."""

    def test_corrcoef_zero_stddev(self):
        torch = pytest.importorskip("torch")
        x = torch.zeros(50, 3)  # all constant columns
        c = corrcoef_pt(x)
        assert torch.isfinite(c).all()

    def test_cov_single_sample(self):
        torch = pytest.importorskip("torch")
        x = torch.randn(1, 3)
        c = cov_pt(x)
        assert torch.isfinite(c).all()


class TestDCIEdgeCases:
    """DCI entropy edge cases."""

    def test_zero_importance_matrix(self):
        R = np.zeros((5, 3))
        d = disentanglement(R)
        c = completeness(R)
        assert 0 <= d <= 1
        assert 0 <= c <= 1

    def test_near_zero_importance_matrix(self):
        R = np.full((5, 3), 1e-15)
        d = disentanglement(R)
        c = completeness(R)
        assert 0 <= d <= 1
        assert 0 <= c <= 1

    def test_disentanglement_per_code_clamped(self):
        R = np.eye(3) * 0.99 + 0.001
        scores = disentanglement_per_code(R)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_completeness_per_factor_clamped(self):
        R = np.eye(3) * 0.99 + 0.001
        scores = completeness_per_factor(R)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestR2NearConstant:
    """R2 with near-constant target must not blow up."""

    def test_near_constant_target(self):
        np.random.seed(42)
        Z = np.column_stack([
            np.ones(100) + 1e-20 * np.random.randn(100),
            np.random.randn(100),
        ])
        Z_hat = np.random.randn(100, 3)
        r2 = R2Metric()
        result = r2.compute(Z, Z_hat)
        assert np.isfinite(result.primary_score)

    def test_constant_target(self):
        Z = np.ones((100, 2))
        Z_hat = np.random.randn(100, 3)
        r2 = R2Metric()
        result = r2.compute(Z, Z_hat)
        assert np.isfinite(result.primary_score)


class TestInfoMECConstantFactor:
    """InfoMEC with a constant factor."""

    def test_constant_source_factor(self):
        np.random.seed(42)
        Z = np.column_stack([
            np.ones(200),  # constant factor
            np.random.randn(200),
            np.random.randn(200),
        ])
        Z_hat = np.random.randn(200, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infom = InfoM()
            result = infom.compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1


class TestTMEXDegenerate:
    """TMEX with degenerate inputs must not crash."""

    def test_identical_factors(self):
        np.random.seed(42)
        Z = np.random.randn(100, 2)
        Z_hat = Z.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmex = TMEX(regression_method="lm", rep=3, seed=42)
            result = tmex.compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1
