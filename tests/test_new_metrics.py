"""Tests for new Identifiability Metrics (MIG, T-MEX, InfoMEC)."""

import numpy as np
import pytest
import warnings

from identifiability_guard.metrics import MIG, TMEX, InfoM, InfoE, InfoC, InfoMECMetric, MetricRegistry


class TestMIG:
    """Tests for Mutual Information Gap metric."""
    
    def test_basic_computation(self):
        """Test MIG can be computed."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z + 0.1 * np.random.randn(500, 3)
        
        mig = MIG()
        result = mig.compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1
    
    def test_perfect_correspondence(self):
        """Test MIG for nearly perfect correspondence."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z.copy()  # Perfect
        
        mig = MIG()
        result = mig.compute(Z, Z_hat)
        # Should be relatively high for perfect correspondence
        assert result.primary_score > 0.3
    
    def test_random_correspondence(self):
        """Test MIG for random representations."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = np.random.randn(500, 3)  # Random
        
        mig = MIG()
        result = mig.compute(Z, Z_hat)
        # Should be low for random
        assert result.primary_score < 0.5
    
    def test_num_bins_parameter(self):
        """Test that num_bins parameter works."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z + 0.1 * np.random.randn(500, 3)
        
        mig_10 = MIG(num_bins=10)
        mig_30 = MIG(num_bins=30)
        
        result_10 = mig_10.compute(Z, Z_hat)
        result_30 = mig_30.compute(Z, Z_hat)
        
        # Both should be valid scores
        assert 0 <= result_10.primary_score <= 1
        assert 0 <= result_30.primary_score <= 1


class TestTMEX:
    """Tests for T-MEX metric."""
    
    def test_basic_computation(self):
        """Test T-MEX can be computed."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z + 0.1 * np.random.randn(200, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmex = TMEX(regression_method="lm", rep=3, seed=42)
            result = tmex.compute(Z, Z_hat)
        
        assert 0 <= result.primary_score <= 1
    
    def test_perfect_correspondence(self):
        """Test T-MEX for nearly perfect correspondence."""
        np.random.seed(42)
        Z = np.random.randn(300, 3)
        Z_hat = Z + 0.05 * np.random.randn(300, 3)  # Very close
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmex = TMEX(regression_method="lm", rep=5, seed=42)
            result = tmex.compute(Z, Z_hat)
        
        # Should detect correspondence
        assert result.primary_score >= 0.5
    
    def test_metadata(self):
        """Test that metadata is populated."""
        np.random.seed(42)
        Z = np.random.randn(150, 3)
        Z_hat = Z + 0.1 * np.random.randn(150, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmex = TMEX(regression_method="lm", rep=3, seed=42)
            result = tmex.compute(Z, Z_hat)
        
        assert "correspondence_matrix" in result.metadata
        assert "regression_method" in result.metadata
        assert "alpha" in result.metadata


class TestInfoMEC:
    """Tests for InfoMEC metrics (InfoM, InfoE, InfoC)."""
    
    def test_infom_basic(self):
        """Test InfoM can be computed."""
        np.random.seed(42)
        Z = np.random.randn(300, 3)
        Z_hat = Z + 0.1 * np.random.randn(300, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infom = InfoM()
            result = infom.compute(Z, Z_hat)
        
        assert 0 <= result.primary_score <= 1
    
    def test_infoe_basic(self):
        """Test InfoE can be computed."""
        np.random.seed(42)
        Z = np.random.randn(300, 3)
        Z_hat = Z + 0.1 * np.random.randn(300, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infoe = InfoE()
            result = infoe.compute(Z, Z_hat)
        
        assert 0 <= result.primary_score <= 1
    
    def test_infoc_basic(self):
        """Test InfoC can be computed."""
        np.random.seed(42)
        Z = np.random.randn(300, 3)
        Z_hat = Z + 0.1 * np.random.randn(300, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infoc = InfoC()
            result = infoc.compute(Z, Z_hat)
        
        assert 0 <= result.primary_score <= 1
    
    def test_infomec_combined(self):
        """Test InfoMEC combined metric."""
        np.random.seed(42)
        Z = np.random.randn(300, 3)
        Z_hat = Z + 0.1 * np.random.randn(300, 3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infomec = InfoMECMetric()
            result = infomec.compute(Z, Z_hat)
        
        assert 0 <= result.primary_score <= 1
        assert "modularity" in result.subscores
        assert "explicitness" in result.subscores
        assert "compactness" in result.subscores
    
    def test_disentangled_high_scores(self):
        """Test that disentangled representations get high scores."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z + 0.01 * np.random.randn(500, 3)  # Very close to identity
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infom = InfoM()
            infoc = InfoC()
            
            infom_result = infom.compute(Z, Z_hat)
            infoc_result = infoc.compute(Z, Z_hat)
        
        # Should be high for disentangled representation
        assert infom_result.primary_score > 0.5
        assert infoc_result.primary_score > 0.5


class TestMetricRegistry:
    """Test that new metrics are properly registered."""
    
    def test_new_metrics_registered(self):
        """Test that new metrics appear in registry."""
        registry = MetricRegistry()
        registry.register_defaults()
        
        metrics = registry.list_metrics()
        assert "mig" in metrics
        assert "tmex" in metrics
        assert "infom" in metrics
        assert "infoe" in metrics
        assert "infoc" in metrics
    
    def test_compute_all_includes_new_metrics(self):
        """Test that compute_all includes new metrics."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z + 0.1 * np.random.randn(200, 3)
        
        registry = MetricRegistry()
        registry.register_defaults()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = registry.compute_all(Z, Z_hat)
        
        assert "mig" in results
        assert "tmex" in results
        assert "infom" in results
        assert "infoe" in results
        assert "infoc" in results
