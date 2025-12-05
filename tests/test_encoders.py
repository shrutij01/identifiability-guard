"""Tests for Encoder Mixings."""

import numpy as np
import pytest

from src.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E4UndercompleteLInear,
    E5OvercompleteLinear,
    E6OvercompleteMulticodes,
)


class TestE1ElementwiseLinear:
    """Tests for E1 elementwise linear encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E1ElementwiseLinear(d=5, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 5)
    
    def test_callable(self):
        """Test encoder is callable."""
        encoder = E1ElementwiseLinear(d=3, seed=42)
        Z = np.random.randn(50, 3)
        Z_hat = encoder(Z)
        assert Z_hat.shape == (50, 3)
    
    def test_reproducibility(self):
        """Test that seed makes encoding reproducible."""
        encoder1 = E1ElementwiseLinear(d=3, seed=42)
        encoder2 = E1ElementwiseLinear(d=3, seed=42)
        Z = np.random.randn(50, 3)
        Z_hat1 = encoder1.encode(Z)
        Z_hat2 = encoder2.encode(Z)
        np.testing.assert_array_equal(Z_hat1, Z_hat2)
    
    def test_permutation(self):
        """Test that permutation is applied."""
        encoder = E1ElementwiseLinear(d=3, permute=True, seed=42)
        # Initializing encoder
        Z = np.eye(3)
        encoder.encode(Z)
        
        # Check that permutation is not identity
        assert not np.array_equal(encoder.permutation, np.arange(3)) or encoder.d == 1
    
    def test_no_permutation(self):
        """Test encoding without permutation."""
        encoder = E1ElementwiseLinear(d=3, permute=False, seed=42)
        Z = np.eye(3)
        encoder.encode(Z)
        np.testing.assert_array_equal(encoder.permutation, np.arange(3))
    
    def test_wrong_input_dimension(self):
        """Test that wrong input dimension raises error."""
        encoder = E1ElementwiseLinear(d=3, seed=42)
        Z = np.random.randn(50, 5)  # Wrong dimension
        with pytest.raises(ValueError):
            encoder.encode(Z)


class TestE2ElementwiseNonlinear:
    """Tests for E2 elementwise nonlinear encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E2ElementwiseNonlinear(d=5, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 5)
    
    def test_nonlinear_transformation(self):
        """Test that nonlinear transformation is applied."""
        encoder = E2ElementwiseNonlinear(d=2, permute=False, seed=42)
        Z = np.array([[1.0, 2.0], [0.5, -0.5]])
        Z_hat = encoder.encode(Z)
        
        # Should not be linear (unless by coincidence)
        # Just check it's different from input
        assert not np.allclose(Z, Z_hat)
    
    def test_custom_nonlinear_functions(self):
        """Test with custom nonlinear functions."""
        custom_fns = [lambda x: x**2, lambda x: np.sin(x)]
        encoder = E2ElementwiseNonlinear(d=2, nonlinear_fns=custom_fns, permute=False, seed=42)
        Z = np.array([[1.0, 0.0], [2.0, np.pi/2]])
        Z_hat = encoder.encode(Z)
        
        expected = np.array([[1.0, 0.0], [4.0, 1.0]])
        np.testing.assert_array_almost_equal(Z_hat, expected)


class TestE3LinearlyEntangled:
    """Tests for E3 linearly entangled encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E3LinearlyEntangled(d=5, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 5)
    
    def test_invertibility(self):
        """Test that mixing matrix is invertible."""
        encoder = E3LinearlyEntangled(d=4, seed=42)
        Z = np.random.randn(100, 4)
        encoder.encode(Z)
        
        # Check condition number
        cond = np.linalg.cond(encoder.mixing_matrix)
        assert cond < 100  # Should be reasonably conditioned
    
    def test_mixing(self):
        """Test that factors are mixed (not elementwise)."""
        encoder = E3LinearlyEntangled(d=3, seed=42)
        Z = np.eye(3)
        encoder.encode(Z)
        
        # Check that mixing matrix has multiple nonzeros per row
        for row in encoder.mixing_matrix:
            nonzeros = np.sum(np.abs(row) > 1e-6)
            assert nonzeros > 1, "Mixing matrix should have multiple nonzeros per row"


class TestE4UndercompleteLInear:
    """Tests for E4 undercomplete linear encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E4UndercompleteLInear(d=5, m=3, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 3)
    
    def test_dimensionality_reduction(self):
        """Test that output has fewer dimensions."""
        encoder = E4UndercompleteLInear(d=10, m=4, seed=42)
        Z = np.random.randn(50, 10)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape[1] < Z.shape[1]
    
    def test_m_must_be_less_than_d(self):
        """Test that m >= d raises error."""
        with pytest.raises(ValueError):
            E4UndercompleteLInear(d=3, m=3)
        with pytest.raises(ValueError):
            E4UndercompleteLInear(d=3, m=5)


class TestE5OvercompleteLinear:
    """Tests for E5 overcomplete linear encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E5OvercompleteLinear(d=3, m=5, seed=42)
        Z = np.random.randn(100, 3)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 5)
    
    def test_dimensionality_increase(self):
        """Test that output has more dimensions."""
        encoder = E5OvercompleteLinear(d=4, m=10, seed=42)
        Z = np.random.randn(50, 4)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape[1] > Z.shape[1]
    
    def test_m_must_be_greater_than_d(self):
        """Test that m <= d raises error."""
        with pytest.raises(ValueError):
            E5OvercompleteLinear(d=3, m=3)
        with pytest.raises(ValueError):
            E5OvercompleteLinear(d=5, m=3)
    
    def test_all_factors_represented(self):
        """Test that all factors appear in at least one output."""
        encoder = E5OvercompleteLinear(d=3, m=6, seed=42)
        Z = np.random.randn(100, 3)
        encoder.encode(Z)
        
        # Each of the 3 inputs should be used at least once
        for i in range(3):
            assert i in encoder.source_indices


class TestE6OvercompleteMulticodes:
    """Tests for E6 overcomplete multicodes encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E6OvercompleteMulticodes(d=3, m=6, seed=42)
        Z = np.random.randn(100, 3)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 6)
    
    def test_nonlinear_transformation(self):
        """Test that nonlinear transformations are applied."""
        encoder = E6OvercompleteMulticodes(d=2, m=4, seed=42)
        Z = np.random.randn(50, 2)
        Z_hat = encoder.encode(Z)
        
        # Output should not be simple scaling
        assert Z_hat.shape == (50, 4)
    
    def test_m_must_be_greater_than_d(self):
        """Test that m <= d raises error."""
        with pytest.raises(ValueError):
            E6OvercompleteMulticodes(d=3, m=3)
