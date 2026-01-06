"""
Timing and memory profiling utilities for evaluation.

Provides context managers and decorators for measuring execution time 
and memory usage of evaluation tasks.
"""

import time
import tracemalloc
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Optional, Any, Tuple
import sys


@contextmanager
def time_block(name: str = "Block", verbose: bool = True):
    """
    Context manager for timing a code block.
    
    Args:
        name: Name of the block being timed.
        verbose: If True, prints timing information.
        
    Yields:
        dict: Dictionary to store timing results with key 'elapsed'.
        
    Example:
        >>> with time_block("Data generation") as timer:
        ...     data = generate_data(n=10000)
        >>> print(f"Took {timer['elapsed']:.3f} seconds")
    """
    result = {}
    start = time.perf_counter()
    
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result['elapsed'] = elapsed
        if verbose:
            print(f"[Timing] {name}: {elapsed:.3f} seconds")


@contextmanager
def memory_profiler(name: str = "Block", verbose: bool = True):
    """
    Context manager for profiling memory usage of a code block.
    
    Uses tracemalloc from the standard library to measure memory allocation.
    
    Args:
        name: Name of the block being profiled.
        verbose: If True, prints memory usage information.
        
    Yields:
        dict: Dictionary with keys 'current', 'peak' (in MB).
        
    Example:
        >>> with memory_profiler("Model training") as mem:
        ...     model.fit(X, y)
        >>> print(f"Peak memory: {mem['peak']:.2f} MB")
    """
    result = {}
    
    # Start tracing
    tracemalloc.start()
    
    try:
        yield result
    finally:
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        result['current'] = current_mb
        result['peak'] = peak_mb
        
        if verbose:
            print(f"[Memory] {name}: Current={current_mb:.2f} MB, Peak={peak_mb:.2f} MB")


@contextmanager
def profile_block(name: str = "Block", verbose: bool = True):
    """
    Context manager for combined timing and memory profiling.
    
    Args:
        name: Name of the block being profiled.
        verbose: If True, prints timing and memory information.
        
    Yields:
        dict: Dictionary with keys 'elapsed', 'current_mb', 'peak_mb'.
        
    Example:
        >>> with profile_block("Full evaluation") as profile:
        ...     run_evaluation(data)
        >>> print(f"Time: {profile['elapsed']:.2f}s, Peak Memory: {profile['peak_mb']:.2f} MB")
    """
    result = {}
    
    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    try:
        yield result
    finally:
        # Get metrics
        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Store results
        result['elapsed'] = elapsed
        result['current_mb'] = current / 1024 / 1024
        result['peak_mb'] = peak / 1024 / 1024
        
        if verbose:
            print(f"[Profile] {name}:")
            print(f"  Time: {elapsed:.3f} seconds")
            print(f"  Memory: Current={result['current_mb']:.2f} MB, Peak={result['peak_mb']:.2f} MB")


def timed(verbose: bool = True):
    """
    Decorator for timing function execution.
    
    Args:
        verbose: If True, prints timing information.
        
    Returns:
        Decorated function that records execution time.
        
    Example:
        >>> @timed()
        ... def compute_metrics(data):
        ...     return process(data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if verbose:
                print(f"[Timing] {func.__name__}: {elapsed:.3f} seconds")
            
            # Store timing info as attribute
            if not hasattr(wrapper, '_timing_history'):
                wrapper._timing_history = []
            wrapper._timing_history.append(elapsed)
            
            return result
        return wrapper
    return decorator


def profiled(verbose: bool = True):
    """
    Decorator for combined timing and memory profiling of functions.
    
    Args:
        verbose: If True, prints profiling information.
        
    Returns:
        Decorated function that records time and memory usage.
        
    Example:
        >>> @profiled()
        ... def train_model(X, y):
        ...     return fit(X, y)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start = time.perf_counter()
            
            result = func(*args, **kwargs)
            
            elapsed = time.perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            profile_info = {
                'elapsed': elapsed,
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024,
            }
            
            if verbose:
                print(f"[Profile] {func.__name__}:")
                print(f"  Time: {elapsed:.3f} seconds")
                print(f"  Memory: Peak={profile_info['peak_mb']:.2f} MB")
            
            # Store profile info
            if not hasattr(wrapper, '_profile_history'):
                wrapper._profile_history = []
            wrapper._profile_history.append(profile_info)
            
            return result
        return wrapper
    return decorator


class Timer:
    """
    Simple timer class for manual timing.
    
    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do work ...
        >>> elapsed = timer.stop()
        >>> print(f"Elapsed: {elapsed:.3f}s")
    """
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._elapsed = None
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds.
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")
        
        self._elapsed = time.perf_counter() - self._start_time
        return self._elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time (either current or last stopped time)."""
        if self._start_time is None:
            raise RuntimeError("Timer not started.")
        
        if self._elapsed is not None:
            return self._elapsed
        
        return time.perf_counter() - self._start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
