"""MCC: Mean Correlation Coefficient metric."""

from itertools import permutations
from typing import Optional

import numpy as np

from .base import BaseMetric


class MCC(BaseMetric):
    """
    Mean Correlation Coefficient (MCC) metric.
    
    MCC(ρ) = (1/k) * max_{π ∈ S_k} Σ_{i=1}^{k} |Corr(Z_i, Ẑ_π(i))|
    
    where k = min(d, m) and we find the best permutation that maximizes
    the average absolute correlation between matched pairs.
    
    For large dimensions, finding the optimal permutation is expensive
    (factorial complexity), so we use the Hungarian algorithm for efficiency.
    """
    
    def __init__(self, use_hungarian: bool = True):
        """
        Initialize the MCC metric.
        
        Args:
            use_hungarian: If True, use Hungarian algorithm for optimal matching.
                If False, try all permutations (only feasible for small d, m).
        """
        self.use_hungarian = use_hungarian
    
    def compute(self, Z: np.ndarray, Z_hat: np.ndarray) -> float:
        """
        Compute the MCC metric.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            Z_hat: Array of shape (n, m) containing learned coordinates.
            
        Returns:
            mcc: Float in [0, 1], higher is better.
        """
        n, d = Z.shape
        _, m = Z_hat.shape
        k = min(d, m)
        
        if n < 2:
            raise ValueError("Need at least 2 samples to compute correlations")
        
        # Compute absolute correlation matrix
        corr_matrix = self._compute_correlation_matrix(Z, Z_hat)
        abs_corr = np.abs(corr_matrix)
        
        # Find optimal matching
        if self.use_hungarian or k > 8:
            # Use Hungarian algorithm (linear sum assignment)
            mcc = self._hungarian_matching(abs_corr)
        else:
            # Brute force all permutations
            mcc = self._brute_force_matching(abs_corr)
        
        return mcc
    
    def _compute_correlation_matrix(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlation matrix between Z and Z_hat.
        
        Args:
            Z: Array of shape (n, d).
            Z_hat: Array of shape (n, m).
            
        Returns:
            corr: Array of shape (d, m) with Pearson correlations.
        """
        d = Z.shape[1]
        m = Z_hat.shape[1]
        
        # Standardize
        Z_std = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-10)
        Z_hat_std = (Z_hat - Z_hat.mean(axis=0)) / (Z_hat.std(axis=0) + 1e-10)
        
        # Compute correlation matrix
        corr = np.zeros((d, m))
        for i in range(d):
            for j in range(m):
                corr[i, j] = np.mean(Z_std[:, i] * Z_hat_std[:, j])
        
        return corr
    
    def _hungarian_matching(self, abs_corr: np.ndarray) -> float:
        """
        Find optimal matching using Hungarian algorithm.
        
        Args:
            abs_corr: Absolute correlation matrix of shape (d, m).
            
        Returns:
            mcc: Mean of matched absolute correlations.
        """
        from scipy.optimize import linear_sum_assignment
        
        d, m = abs_corr.shape
        k = min(d, m)
        
        # Hungarian algorithm minimizes, so we negate for maximization
        cost_matrix = -abs_corr
        
        # Handle non-square matrices
        if d != m:
            # Pad with zeros
            max_dim = max(d, m)
            padded = np.zeros((max_dim, max_dim))
            padded[:d, :m] = cost_matrix
            cost_matrix = padded
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute mean of matched correlations (only for valid pairs)
        total_corr = 0.0
        count = 0
        for i, j in zip(row_ind, col_ind):
            if i < d and j < m:
                total_corr += abs_corr[i, j]
                count += 1
        
        return total_corr / k if k > 0 else 0.0
    
    def _brute_force_matching(self, abs_corr: np.ndarray) -> float:
        """
        Find optimal matching by trying all permutations.
        
        Args:
            abs_corr: Absolute correlation matrix of shape (d, m).
            
        Returns:
            mcc: Mean of matched absolute correlations.
        """
        d, m = abs_corr.shape
        k = min(d, m)
        
        best_mcc = 0.0
        
        if d <= m:
            # Iterate over permutations of columns (m choose d, then permute)
            for perm in permutations(range(m), d):
                total = sum(abs_corr[i, perm[i]] for i in range(d))
                mcc = total / d
                best_mcc = max(best_mcc, mcc)
        else:
            # Iterate over permutations of rows (d choose m, then permute)
            for perm in permutations(range(d), m):
                total = sum(abs_corr[perm[j], j] for j in range(m))
                mcc = total / m
                best_mcc = max(best_mcc, mcc)
        
        return best_mcc
