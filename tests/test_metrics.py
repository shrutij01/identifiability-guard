"""Tests for Identifiability Metrics."""

import numpy as np
import pytest

from src.metrics import MCC, DCI


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
