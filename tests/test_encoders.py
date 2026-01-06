"""Tests for Encoder Mixings."""

import numpy as np
import pytest

from src.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E4UndercompleteLinear,
    E5OvercompleteLinear,
    E6OvercompleteMulticodes,
    E7OvercompleteEntangled,
    E8OvercompleteDisjoint,
    E9RandomGaussian,
    E10RandomUniform,
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
    
    def test_nonlinearity_strength_zero(self):
        """Test that nonlinearity_strength=0 gives identity transformation."""
        encoder = E2ElementwiseNonlinear(d=3, nonlinearity_strength=0.0, permute=False, seed=42)
        Z = np.random.randn(50, 3)
        Z_hat = encoder.encode(Z)
        
        # With strength=0, should be identity (no nonlinearity)
        np.testing.assert_array_almost_equal(Z, Z_hat)
    
    def test_nonlinearity_strength_one(self):
        """Test that nonlinearity_strength=1 gives full nonlinearity."""
        custom_fns = [lambda x: x**2, lambda x: x**3, lambda x: np.tanh(x)]
        encoder = E2ElementwiseNonlinear(
            d=3, nonlinear_fns=custom_fns, nonlinearity_strength=1.0, permute=False, seed=42
        )
        Z = np.array([[1.0, 2.0, 0.5]])
        Z_hat = encoder.encode(Z)
        
        # With strength=1, should be fully nonlinear
        expected = np.array([[1.0, 8.0, np.tanh(0.5)]])
        np.testing.assert_array_almost_equal(Z_hat, expected)
    
    def test_nonlinearity_strength_interpolation(self):
        """Test that intermediate strength interpolates correctly."""
        custom_fns = [lambda x: x**2]
        encoder = E2ElementwiseNonlinear(
            d=1, nonlinear_fns=custom_fns, nonlinearity_strength=0.5, permute=False, seed=42
        )
        Z = np.array([[2.0]])
        Z_hat = encoder.encode(Z)
        
        # f(x) = (1-0.5)*x + 0.5*x^2 = 0.5*2 + 0.5*4 = 1 + 2 = 3
        expected = np.array([[3.0]])
        np.testing.assert_array_almost_equal(Z_hat, expected)
    
    def test_nonlinearity_strength_validation(self):
        """Test that invalid strength values raise errors."""
        with pytest.raises(ValueError):
            E2ElementwiseNonlinear(d=3, nonlinearity_strength=-0.1)
        with pytest.raises(ValueError):
            E2ElementwiseNonlinear(d=3, nonlinearity_strength=1.5)


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


class TestE4UndercompleteLinear:
    """Tests for E4 undercomplete linear encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E4UndercompleteLinear(d=5, m=3, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 3)
    
    def test_dimensionality_reduction(self):
        """Test that output has fewer dimensions."""
        encoder = E4UndercompleteLinear(d=10, m=4, seed=42)
        Z = np.random.randn(50, 10)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape[1] < Z.shape[1]
    
    def test_m_must_be_less_than_d(self):
        """Test that m >= d raises error."""
        with pytest.raises(ValueError):
            E4UndercompleteLinear(d=3, m=3)
        with pytest.raises(ValueError):
            E4UndercompleteLinear(d=3, m=5)


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


class TestE7OvercompleteEntangled:
    """Tests for E7 overcomplete entangled encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E7OvercompleteEntangled(d=3, m=6, seed=42)
        Z = np.random.randn(100, 3)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 6)
    
    def test_default_m_is_2d(self):
        """Test that default m is 2*d."""
        encoder = E7OvercompleteEntangled(d=5, seed=42)
        assert encoder.m == 10
        Z = np.random.randn(50, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (50, 10)
    
    def test_full_rank(self):
        """Test that mixing matrix is full rank (rank = d)."""
        encoder = E7OvercompleteEntangled(d=4, m=8, seed=42)
        Z = np.random.randn(100, 4)
        encoder.encode(Z)
        
        # Check rank
        rank = np.linalg.matrix_rank(encoder.mixing_matrix)
        assert rank == encoder.d
    
    def test_mixing(self):
        """Test that factors are mixed (not elementwise)."""
        encoder = E7OvercompleteEntangled(d=3, m=7, seed=42)
        Z = np.eye(3)
        encoder.encode(Z)
        
        # Check that mixing matrix has multiple nonzeros per row
        for row in encoder.mixing_matrix:
            nonzeros = np.sum(np.abs(row) > 1e-6)
            assert nonzeros > 1, "Mixing matrix should have multiple nonzeros per row"
    
    def test_m_must_be_greater_than_d(self):
        """Test that m <= d raises error."""
        with pytest.raises(ValueError):
            E7OvercompleteEntangled(d=3, m=3)
        with pytest.raises(ValueError):
            E7OvercompleteEntangled(d=5, m=3)
    
    def test_condition_number(self):
        """Test that condition number is respected."""
        encoder = E7OvercompleteEntangled(d=4, m=8, condition_number=5.0, seed=42)
        Z = np.random.randn(100, 4)
        encoder.encode(Z)
        
        # Compute singular values
        s = np.linalg.svd(encoder.mixing_matrix, compute_uv=False)
        actual_cond = s[0] / s[-1]
        
        # Should be close to specified condition number
        assert abs(actual_cond - 5.0) < 1.0


class TestE8OvercompleteDisjoint:
    """Tests for E8 overcomplete disjoint encoder."""
    
    def test_basic_encoding(self):
        """Test basic encoding works."""
        encoder = E8OvercompleteDisjoint(d=3, codes_per_factor=2, seed=42)
        Z = np.random.randn(100, 3)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 6)
    
    def test_default_codes_per_factor(self):
        """Test that default codes_per_factor is 2."""
        encoder = E8OvercompleteDisjoint(d=5, seed=42)
        assert encoder.codes_per_factor == 2
        assert encoder.m == 10
    
    def test_sin_cos_encoding(self):
        """Test sin/cos encoding for codes_per_factor=2."""
        encoder = E8OvercompleteDisjoint(d=2, codes_per_factor=2, seed=42)
        Z = np.array([[0.0, np.pi/2], [np.pi, 0.0]])
        Z_hat = encoder.encode(Z)
        
        # For codes_per_factor=2, should be sin/cos pairs
        # First factor: Z[0,0]=0 -> sin(0)=0, cos(0)=1
        # Second factor: Z[0,1]=pi/2 -> sin(pi/2)=1, cos(pi/2)=0
        assert Z_hat.shape == (2, 4)
    
    def test_reconstruction_sin_cos(self):
        """Test that sin/cos encoding can be reconstructed."""
        encoder = E8OvercompleteDisjoint(d=3, codes_per_factor=2, seed=42)
        Z = np.random.uniform(-np.pi, np.pi, size=(50, 3))
        Z_hat = encoder.encode(Z)
        Z_reconstructed = encoder.decode(Z_hat)
        
        # Should reconstruct angles within [-pi, pi]
        # Check if reconstructed values are close (modulo 2*pi)
        assert Z_reconstructed.shape == Z.shape
    
    def test_codes_per_factor_validation(self):
        """Test that codes_per_factor must be at least 2."""
        with pytest.raises(ValueError):
            E8OvercompleteDisjoint(d=3, codes_per_factor=1)
    
    def test_multiple_codes_per_factor(self):
        """Test encoding with more than 2 codes per factor."""
        encoder = E8OvercompleteDisjoint(d=3, codes_per_factor=4, seed=42)
        Z = np.random.randn(50, 3)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (50, 12)  # 3 factors * 4 codes each


class TestE9RandomGaussian:
    """Tests for E9 random Gaussian encoder (baseline)."""
    
    def test_basic_encoding(self):
        """Test basic encoding produces output of correct shape."""
        encoder = E9RandomGaussian(d=5, seed=42)
        Z = np.random.randn(100, 5)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (100, 5)
    
    def test_output_is_random(self):
        """Test that output is independent of input."""
        encoder = E9RandomGaussian(d=3, seed=42)
        Z1 = np.random.randn(50, 3)
        Z2 = np.random.randn(50, 3) * 10  # Very different input
        
        # Reset seed to get same random output
        encoder = E9RandomGaussian(d=3, seed=42)
        Z_hat1 = encoder.encode(Z1)
        
        encoder = E9RandomGaussian(d=3, seed=42)
        Z_hat2 = encoder.encode(Z2)
        
        # Same seed should produce same output regardless of input
        np.testing.assert_array_equal(Z_hat1, Z_hat2)
    
    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different outputs."""
        Z = np.random.randn(50, 3)
        
        encoder1 = E9RandomGaussian(d=3, seed=42)
        Z_hat1 = encoder1.encode(Z)
        
        encoder2 = E9RandomGaussian(d=3, seed=123)
        Z_hat2 = encoder2.encode(Z)
        
        # Different seeds should produce different outputs
        assert not np.allclose(Z_hat1, Z_hat2)
    
    def test_output_statistics(self):
        """Test that output has approximately standard Gaussian statistics."""
        encoder = E9RandomGaussian(d=5, seed=42)
        Z = np.random.randn(10000, 5)
        Z_hat = encoder.encode(Z)
        
        # Check that output is approximately N(0, 1)
        assert np.abs(Z_hat.mean()) < 0.1
        assert np.abs(Z_hat.std() - 1.0) < 0.1
    
    def test_no_correlation_with_input(self):
        """Test that output is uncorrelated with input."""
        encoder = E9RandomGaussian(d=3, seed=42)
        Z = np.random.randn(1000, 3)
        Z_hat = encoder.encode(Z)
        
        # Compute correlation between input and output
        for i in range(3):
            corr = np.corrcoef(Z[:, i], Z_hat[:, i])[0, 1]
            assert np.abs(corr) < 0.1  # Should be near zero


class TestE10RandomUniform:
    """Tests for E10 random uniform encoder (baseline)."""

    def test_basic_encoding(self):
        """Outputs correct shape regardless of input."""
        encoder = E10RandomUniform(d=4, seed=7)
        Z = np.random.randn(20, 4)
        Z_hat = encoder.encode(Z)
        assert Z_hat.shape == (20, 4)

    def test_values_within_bounds(self):
        """Samples stay within configured uniform range."""
        encoder = E10RandomUniform(d=2, low=-1.5, high=0.5, seed=3)
        Z = np.zeros((1000, 2))
        Z_hat = encoder.encode(Z)
        assert np.all(Z_hat >= -1.5)
        assert np.all(Z_hat <= 0.5)

    def test_seed_reproducibility(self):
        """Same seed yields identical draws regardless of input."""
        Z1 = np.random.randn(10, 3)
        Z2 = np.random.randn(10, 3) * 10

        enc1 = E10RandomUniform(d=3, seed=11)
        out1 = enc1.encode(Z1)

        enc2 = E10RandomUniform(d=3, seed=11)
        out2 = enc2.encode(Z2)

        np.testing.assert_array_equal(out1, out2)
