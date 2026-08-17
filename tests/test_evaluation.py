"""Tests for evaluation utilities."""

import pytest
import numpy as np
import time
from pathlib import Path
import tempfile
import json

from identifiability_guard.evaluation.timing import time_block, Timer, memory_profiler
from identifiability_guard.evaluation.multi_seed import (
    run_with_seeds,
    compute_statistics,
    aggregate_results,
    run_multi_seed_evaluation,
)
from identifiability_guard.evaluation.sensitivity import (
    parameter_sweep,
    sensitivity_analysis_1d,
    compute_sensitivity_statistics,
    save_sensitivity_results,
    load_sensitivity_results,
)


class TestTiming:
    """Tests for timing utilities."""
    
    def test_time_block(self):
        """Test time_block context manager."""
        with time_block("test", verbose=False) as timer:
            time.sleep(0.01)
        
        assert 'elapsed' in timer
        assert timer['elapsed'] >= 0.01
    
    def test_timer_class(self):
        """Test Timer class."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        
        assert elapsed >= 0.01
        assert timer.elapsed >= 0.01
    
    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        with Timer() as timer:
            time.sleep(0.01)
        
        assert timer.elapsed >= 0.01
    
    def test_memory_profiler(self):
        """Test memory profiler context manager."""
        with memory_profiler("test", verbose=False) as mem:
            # Allocate some memory
            data = np.random.randn(1000, 1000)
        
        assert 'current' in mem
        assert 'peak' in mem
        assert mem['peak'] > 0


class TestMultiSeed:
    """Tests for multi-seed evaluation utilities."""
    
    def test_run_with_seeds(self):
        """Test run_with_seeds function."""
        def eval_fn(seed):
            np.random.seed(seed)
            return {"metric1": np.random.rand(), "metric2": np.random.rand()}
        
        results = run_with_seeds(eval_fn, seeds=[42, 43, 44], verbose=False)
        
        assert "metric1" in results
        assert "metric2" in results
        assert len(results["metric1"]) == 3
        assert len(results["metric2"]) == 3
    
    def test_compute_statistics(self):
        """Test compute_statistics function."""
        values = [0.8, 0.85, 0.82, 0.88, 0.84]
        stats = compute_statistics(values, confidence_level=0.95)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'ci_lower' in stats
        assert 'ci_upper' in stats
        assert abs(stats['mean'] - np.mean(values)) < 1e-6
        assert stats['ci_lower'] < stats['mean'] < stats['ci_upper']
    
    def test_aggregate_results(self):
        """Test aggregate_results function."""
        results = {
            "metric1": [0.8, 0.85, 0.82],
            "metric2": [0.9, 0.92, 0.88],
        }
        
        aggregated = aggregate_results(results)
        
        assert "metric1" in aggregated
        assert "metric2" in aggregated
        assert 'mean' in aggregated["metric1"]
        assert 'std' in aggregated["metric1"]
    
    def test_run_multi_seed_evaluation(self):
        """Test run_multi_seed_evaluation function."""
        def eval_fn(seed):
            np.random.seed(seed)
            return {"score": 0.8 + 0.05 * np.random.randn()}
        
        raw, agg = run_multi_seed_evaluation(
            eval_fn,
            n_seeds=3,
            base_seed=42,
            verbose=False,
        )
        
        assert "score" in raw
        assert "score" in agg
        assert len(raw["score"]) == 3
        assert 'mean' in agg["score"]


class TestSensitivity:
    """Tests for sensitivity analysis utilities."""
    
    def test_parameter_sweep(self):
        """Test parameter_sweep function."""
        def eval_fn(params):
            return {"score": params['x'] + params['y']}
        
        results = parameter_sweep(
            eval_fn,
            param_grid={"x": [1, 2], "y": [3, 4]},
            verbose=False,
        )
        
        assert len(results) == 4  # 2 * 2 combinations
        assert all('params' in r and 'metrics' in r for r in results)
    
    def test_sensitivity_analysis_1d(self):
        """Test sensitivity_analysis_1d function."""
        def eval_fn(params):
            return {"score": params['x'] ** 2}
        
        values, results = sensitivity_analysis_1d(
            eval_fn,
            param_name='x',
            param_values=[1, 2, 3],
            n_seeds=2,
            verbose=False,
        )
        
        assert values == [1, 2, 3]
        assert "score" in results
        assert len(results["score"]) == 3
        assert all(len(seed_results) == 2 for seed_results in results["score"])
    
    def test_compute_sensitivity_statistics(self):
        """Test compute_sensitivity_statistics function."""
        param_values = [1, 2, 3]
        metric_results = {
            "score": [[0.8, 0.82], [0.85, 0.87], [0.9, 0.92]]
        }
        
        stats = compute_sensitivity_statistics(param_values, metric_results)
        
        assert "score" in stats
        assert 'mean' in stats["score"]
        assert 'std' in stats["score"]
        assert len(stats["score"]['mean']) == 3
    
    def test_save_load_sensitivity_results(self):
        """Test save and load sensitivity results."""
        results = [
            {'params': {'x': 1}, 'metrics': {'score': 0.8}},
            {'params': {'x': 2}, 'metrics': {'score': 0.85}},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.json"
            
            # Save
            save_sensitivity_results(results, str(filepath))
            assert filepath.exists()
            
            # Load
            loaded_results = load_sensitivity_results(str(filepath))
            assert len(loaded_results) == len(results)
            assert loaded_results[0]['params']['x'] == 1
