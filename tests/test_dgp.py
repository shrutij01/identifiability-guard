"""Tests for Data Generating Processes (DGPs)."""

import numpy as np
import pytest

from src.dgp import D1Independent, D2Correlated, D3SingleRedundant, D4MultiRedundant


class TestD1Independent:
    """Tests for D1 Independent DGP."""
    
    def test_basic_sampling(self):
        """Test basic sampling works."""
        dgp = D1Independent(d=5, seed=42)
        Z = dgp.sample(100)
        assert Z.shape == (100, 5)
    
    def test_reproducibility(self):
        """Test that seed makes sampling reproducible."""
        dgp1 = D1Independent(d=3, seed=42)
        dgp2 = D1Independent(d=3, seed=42)
        Z1 = dgp1.sample(50)
        Z2 = dgp2.sample(50)
        np.testing.assert_array_equal(Z1, Z2)
    
    def test_independence(self):
        """Test that factors are approximately independent."""
        dgp = D1Independent(d=4, seed=42)
        Z = dgp.sample(10000)
        
        # Compute correlation matrix
        corr = np.corrcoef(Z.T)
        
        # Off-diagonal should be close to 0
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(corr[i, j]) < 0.05, f"Correlation [{i},{j}] = {corr[i,j]}"
    
    def test_invalid_d(self):
        """Test that d < 1 raises error."""
        with pytest.raises(ValueError):
            D1Independent(d=0)
    
    def test_invalid_n(self):
        """Test that n < 1 raises error."""
        dgp = D1Independent(d=3)
        with pytest.raises(ValueError):
            dgp.sample(0)


class TestD2Correlated:
    """Tests for D2 Correlated DGP."""
    
    def test_basic_sampling(self):
        """Test basic sampling works."""
        dgp = D2Correlated(d=5, correlation=0.5, seed=42)
        Z = dgp.sample(100)
        assert Z.shape == (100, 5)
    
    def test_correlation_structure(self):
        """Test that factors have expected correlation."""
        target_corr = 0.6
        dgp = D2Correlated(d=3, correlation=target_corr, seed=42)
        Z = dgp.sample(10000)
        
        corr = np.corrcoef(Z.T)
        
        # Off-diagonal should be close to target
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(corr[i, j] - target_corr) < 0.05
    
    def test_custom_correlation_matrix(self):
        """Test custom correlation matrix."""
        custom_corr = np.array([
            [1.0, 0.3, 0.0],
            [0.3, 1.0, 0.5],
            [0.0, 0.5, 1.0]
        ])
        dgp = D2Correlated(d=3, correlation_matrix=custom_corr, seed=42)
        Z = dgp.sample(10000)
        
        corr = np.corrcoef(Z.T)
        np.testing.assert_array_almost_equal(corr, custom_corr, decimal=1)
    
    def test_invalid_correlation(self):
        """Test that invalid correlation raises error."""
        with pytest.raises(ValueError):
            D2Correlated(d=3, correlation=1.5)


class TestD3SingleRedundant:
    """Tests for D3 Single-factor redundant DGP."""
    
    def test_basic_sampling(self):
        """Test basic sampling works."""
        dgp = D3SingleRedundant(d=4, seed=42)
        Z = dgp.sample(100)
        assert Z.shape == (100, 4)
    
    def test_redundancy(self):
        """Test that Z[:, 1] = f(Z[:, 0])."""
        dgp = D3SingleRedundant(d=3, redundant_fn=lambda x: x**2, seed=42)
        Z = dgp.sample(100)
        
        # Z[:, 1] should equal Z[:, 0]**2
        np.testing.assert_array_almost_equal(Z[:, 1], Z[:, 0]**2)
    
    def test_noisy_redundancy(self):
        """Test redundancy with noise."""
        dgp = D3SingleRedundant(d=3, noise_std=0.1, seed=42)
        Z = dgp.sample(1000)
        
        # Correlation should be high but not perfect
        corr = np.corrcoef(Z[:, 0]**2, Z[:, 1])[0, 1]
        assert 0.9 < corr < 1.0
    
    def test_minimum_d(self):
        """Test that d < 2 raises error."""
        with pytest.raises(ValueError):
            D3SingleRedundant(d=1)


class TestD4MultiRedundant:
    """Tests for D4 Multi-factor redundant DGP."""
    
    def test_basic_sampling(self):
        """Test basic sampling works."""
        dgp = D4MultiRedundant(d=5, seed=42)
        Z = dgp.sample(100)
        assert Z.shape == (100, 5)
    
    def test_redundancy(self):
        """Test that Z[:, 2] = g(Z[:, 0], Z[:, 1])."""
        dgp = D4MultiRedundant(d=4, redundant_fn=lambda x, y: x * y, seed=42)
        Z = dgp.sample(100)
        
        # Z[:, 2] should equal Z[:, 0] * Z[:, 1]
        np.testing.assert_array_almost_equal(Z[:, 2], Z[:, 0] * Z[:, 1])
    
    def test_custom_redundancy_function(self):
        """Test with custom redundancy function."""
        dgp = D4MultiRedundant(d=3, redundant_fn=lambda x, y: x + y, seed=42)
        Z = dgp.sample(100)
        
        np.testing.assert_array_almost_equal(Z[:, 2], Z[:, 0] + Z[:, 1])
    
    def test_minimum_d(self):
        """Test that d < 3 raises error."""
        with pytest.raises(ValueError):
            D4MultiRedundant(d=2)
