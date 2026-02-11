"""Tests for Identifiability Metrics."""

import numpy as np
import pytest

from src.metrics import MCC, DCI, MIG, R2, TMEX, InfoMEC, InfoM, InfoE, InfoC


class TestMCC:
    """Tests for Mean Correlation Coefficient metric."""
    
    def test_perfect_correlation(self):
        """Test MCC = 1 for perfect correlation."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z.copy()  # Perfect identity
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        assert score > 0.99
    
    def test_perfect_with_scaling(self):
        """Test MCC = 1 for scaled identity."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z * np.array([2.0, -0.5, 3.0])  # Scaled
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        assert score > 0.99
    
    def test_perfect_with_permutation(self):
        """Test MCC = 1 for permuted identity."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z[:, [2, 0, 1]]  # Permuted
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        assert score > 0.99
    
    def test_uncorrelated(self):
        """Test low MCC for uncorrelated variables."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = np.random.randn(1000, 3)  # Independent
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        assert score < 0.2  # Should be low
    
    def test_dimension_mismatch_more_codes(self):
        """Test MCC with m > d."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = np.column_stack([Z, np.random.randn(1000, 2)])
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        assert score > 0.9  # Should still find good matches
    
    def test_dimension_mismatch_fewer_codes(self):
        """Test MCC with m < d."""
        np.random.seed(42)
        Z = np.random.randn(1000, 5)
        Z_hat = Z[:, :3]  # Only first 3 factors
        
        mcc = MCC()
        score = mcc.compute(Z, Z_hat)
        # Should be high for the 3 matched factors
        assert score > 0.9
    
    def test_brute_force_vs_hungarian(self):
        """Test that brute force and Hungarian give same result."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z[:, [2, 0, 1]] * np.array([1.5, -2.0, 0.5])
        
        mcc_bf = MCC(use_hungarian=False)
        mcc_hung = MCC(use_hungarian=True)
        
        score_bf = mcc_bf.compute(Z, Z_hat)
        score_hung = mcc_hung.compute(Z, Z_hat)
        
        np.testing.assert_almost_equal(score_bf, score_hung, decimal=5)
    
    def test_too_few_samples(self):
        """Test that too few samples raises error."""
        Z = np.random.randn(1, 3)
        Z_hat = np.random.randn(1, 3)
        
        mcc = MCC()
        with pytest.raises(ValueError):
            mcc.compute(Z, Z_hat)
    
    def test_callable(self):
        """Test metric is callable."""
        np.random.seed(42)
        Z = np.random.randn(100, 3)
        Z_hat = Z.copy()
        
        mcc = MCC()
        score = mcc(Z, Z_hat)
        assert 0 <= score <= 1


class TestDCI:
    """Tests for DCI metric."""
    
    def test_basic_computation(self):
        """Test basic DCI computation."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z.copy()
        
        dci = DCI()
        scores = dci.compute(Z, Z_hat)
        
        assert "disentanglement" in scores
        assert "completeness" in scores
        assert "informativeness" in scores
    
    def test_perfect_disentanglement(self):
        """Test high scores for perfect encoding."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z.copy()
        
        dci = DCI()
        scores = dci.compute(Z, Z_hat)
        
        # Perfect encoding should have high scores
        assert scores["disentanglement"] > 0.8
        assert scores["completeness"] > 0.8
        assert scores["informativeness"] > 0.9
    
    def test_entangled_representation(self):
        """Test lower disentanglement for entangled codes."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        # Create entangled representation
        A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        Z_hat = Z @ A.T
        
        dci = DCI()
        scores = dci.compute(Z, Z_hat)
        
        # Entangled should have lower disentanglement
        assert scores["disentanglement"] < 0.8
    
    def test_scores_in_range(self):
        """Test that all scores are in [0, 1]."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = np.random.randn(500, 3)
        
        dci = DCI()
        scores = dci.compute(Z, Z_hat)
        
        for name, score in scores.items():
            assert 0 <= score <= 1, f"{name} = {score} not in [0, 1]"
    
    def test_gradient_boosting_method(self):
        """Test DCI with gradient boosting method."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z.copy()
        
        dci = DCI(method="gradient_boosting", n_estimators=50)
        scores = dci.compute(Z, Z_hat)
        
        assert scores["informativeness"] > 0.8
    
    def test_too_few_samples(self):
        """Test that too few samples raises error."""
        Z = np.random.randn(5, 3)
        Z_hat = np.random.randn(5, 3)
        
        dci = DCI()
        with pytest.raises(ValueError):
            dci.compute(Z, Z_hat)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            DCI(method="invalid")
    
    def test_callable(self):
        """Test metric is callable."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z.copy()
        
        dci = DCI()
        scores = dci(Z, Z_hat)
        assert isinstance(scores, dict)


def _make_data(n=200, d=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d))
    Z_hat = Z + noise * rng.standard_normal((n, d))
    return Z, Z_hat


class TestReproducibility:
    """All metrics should be deterministic when seeded."""

    # -- MCC (RDC uses random projections) --

    def test_mcc_rdc_seeded(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc", seed=42).compute(Z, Z_hat)
        r2 = MCC(method="rdc", seed=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_mcc_rdc_different_seeds(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc", seed=1).compute(Z, Z_hat)
        r2 = MCC(method="rdc", seed=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_mcc_rdc_unseeded_varies(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc").compute(Z, Z_hat)
        r2 = MCC(method="rdc").compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_mcc_pearson_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="pearson").compute(Z, Z_hat)
        r2 = MCC(method="pearson").compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_mcc_spearman_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="spearman").compute(Z, Z_hat)
        r2 = MCC(method="spearman").compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- DCI (train/test split uses random_state) --

    def test_dci_seeded(self):
        Z, Z_hat = _make_data()
        r1 = DCI(random_state=42).compute(Z, Z_hat)
        r2 = DCI(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_dci_different_seeds(self):
        Z, Z_hat = _make_data()
        r1 = DCI(random_state=1).compute(Z, Z_hat)
        r2 = DCI(random_state=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    # -- InfoMEC (MI estimation + logistic regression use randomness) --

    def test_infomec_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoMEC(random_state=42).compute(Z, Z_hat)
        r2 = InfoMEC(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infomec_different_seeds(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoMEC(random_state=1).compute(Z, Z_hat)
        r2 = InfoMEC(random_state=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_infom_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoM(random_state=42).compute(Z, Z_hat)
        r2 = InfoM(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infoe_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoE(random_state=42).compute(Z, Z_hat)
        r2 = InfoE(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infoc_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoC(random_state=42).compute(Z, Z_hat)
        r2 = InfoC(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- TMEX (PCM test uses random splits) --

    def test_tmex_seeded(self):
        Z, Z_hat = _make_data(n=200, d=3)
        r1 = TMEX(seed=42, rep=3).compute(Z, Z_hat)
        r2 = TMEX(seed=42, rep=3).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_tmex_different_seeds(self):
        Z, Z_hat = _make_data(n=200, d=3)
        r1 = TMEX(seed=1, rep=3).compute(Z, Z_hat)
        r2 = TMEX(seed=2, rep=3).compute(Z, Z_hat)
        # TMEX returns binary (0/1) scores so different seeds may still match;
        # we only check that the call succeeds and returns valid scores
        assert 0.0 <= r1.primary_score <= 1.0
        assert 0.0 <= r2.primary_score <= 1.0

    # -- MIG (deterministic: histogram binning + discrete MI) --

    def test_mig_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MIG().compute(Z, Z_hat)
        r2 = MIG().compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- R2 (deterministic: least squares) --

    def test_r2_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = R2().compute(Z, Z_hat)
        r2 = R2().compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score
