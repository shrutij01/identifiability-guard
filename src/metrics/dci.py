"""DCI: Disentanglement, Completeness, Informativeness metric."""

from typing import Dict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

from .base import BaseMetric


class DCI(BaseMetric):
    """
    TODO: use other implem - see with Shruti
    Disentanglement, Completeness, Informativeness (DCI) metric.
    
    The DCI framework measures three aspects:
    
    1. Disentanglement (D): Each learned code should depend on at most one factor.
       D_j = 1 - H(P_{j·}) where P_{ij} = R_{ij} / Σ_i R_{ij}
       D = Σ_j ρ_j D_j where ρ_j = Σ_i R_{ij} / Σ_{ij} R_{ij}
    
    2. Completeness (C): Each factor should be captured by at most one code.
       C_i = 1 - H(P_{·i}) where P_{ij} = R_{ij} / Σ_j R_{ij}
       C = Σ_i ρ_i C_i where ρ_i = Σ_j R_{ij} / Σ_{ij} R_{ij}
    
    3. Informativeness (I): How well codes predict factors (or vice versa).
       Measured as prediction accuracy using gradient boosting.
    
    R is the importance matrix where R_{ij} measures the importance of 
    factor i in predicting code j.
    """
    
    def __init__(
        self,
        method: str = "lasso",
        alpha: float = 0.01,
        n_estimators: int = 100,
    ):
        """
        Initialize the DCI metric.
        
        Args:
            method: Method to compute importance matrix. 
                "lasso": Use Lasso regression coefficients.
                "gradient_boosting": Use gradient boosting feature importances.
            alpha: Regularization strength for Lasso (if method="lasso").
            n_estimators: Number of estimators for gradient boosting.
        """
        if method not in ["lasso", "gradient_boosting"]:
            raise ValueError(f"method must be 'lasso' or 'gradient_boosting', got {method}")
        self.method = method
        self.alpha = alpha
        self.n_estimators = n_estimators
    
    def compute(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute the DCI metrics.
        
        Args:
            Z: Array of shape (n, d) containing ground-truth factors.
            Z_hat: Array of shape (n, m) containing learned coordinates.
            
        Returns:
            dict with keys:
                - "disentanglement": D score in [0, 1]
                - "completeness": C score in [0, 1]
                - "informativeness": I score in [0, 1]
        """
        n, d = Z.shape
        _, m = Z_hat.shape
        
        if n < 10:
            raise ValueError("Need at least 10 samples for DCI computation")
        
        # Compute importance matrix R of shape (d, m)
        # R[i, j] = importance of factor i in predicting code j
        R = self._compute_importance_matrix(Z, Z_hat)
        
        # Compute disentanglement
        disentanglement = self._compute_disentanglement(R)
        
        # Compute completeness
        completeness = self._compute_completeness(R)
        
        # Compute informativeness
        informativeness = self._compute_informativeness(Z, Z_hat)
        
        return {
            "disentanglement": disentanglement,
            "completeness": completeness,
            "informativeness": informativeness,
        }
    
    def _compute_importance_matrix(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> np.ndarray:
        """
        Compute the importance matrix R.
        
        R[i, j] measures how important factor Z_i is for predicting code Z_hat_j.
        
        Args:
            Z: Ground-truth factors of shape (n, d).
            Z_hat: Learned codes of shape (n, m).
            
        Returns:
            R: Importance matrix of shape (d, m).
        """
        d = Z.shape[1]
        m = Z_hat.shape[1]
        R = np.zeros((d, m))
        
        if self.method == "lasso":
            for j in range(m):
                # Predict Z_hat[:, j] from Z
                model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000)
                model.fit(Z, Z_hat[:, j])
                R[:, j] = np.abs(model.coef_)
        else:  # gradient_boosting
            for j in range(m):
                model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators, 
                    max_depth=3,
                    random_state=42
                )
                model.fit(Z, Z_hat[:, j])
                R[:, j] = model.feature_importances_
        
        return R
    
    def _compute_disentanglement(self, R: np.ndarray) -> float:
        """
        Compute disentanglement score.
        
        D_j = 1 - H(P_{j·}) where P_{ij} = R_{ij} / Σ_i R_{ij}
        D = Σ_j ρ_j D_j where ρ_j = Σ_i R_{ij} / Σ_{ij} R_{ij}
        
        Args:
            R: Importance matrix of shape (d, m).
            
        Returns:
            D: Disentanglement score in [0, 1].
        """
        d, m = R.shape
        
        # Normalize importance for each code (column-wise)
        col_sums = R.sum(axis=0) + 1e-10
        P = R / col_sums  # P[i, j] = R[i,j] / Σ_i R[i,j]
        
        # Compute entropy for each code
        H_j = np.zeros(m)
        for j in range(m):
            probs = P[:, j]
            # Entropy: -Σ p log(p), handling zeros
            mask = probs > 1e-10
            if mask.sum() > 0:
                H_j[j] = -np.sum(probs[mask] * np.log(probs[mask]))
        
        # Normalize entropy by log(d) to get values in [0, 1]
        max_entropy = np.log(d) if d > 1 else 1.0
        H_j_normalized = H_j / max_entropy
        
        # Disentanglement per code: D_j = 1 - H_j_normalized
        D_j = 1.0 - H_j_normalized
        
        # Weighted average by code importance
        total_importance = R.sum()
        if total_importance < 1e-10:
            return 0.0
        
        rho_j = col_sums / total_importance
        D = np.sum(rho_j * D_j)
        
        return float(np.clip(D, 0.0, 1.0))
    
    def _compute_completeness(self, R: np.ndarray) -> float:
        """
        Compute completeness score.
        
        C_i = 1 - H(P_{·i}) where P_{ij} = R_{ij} / Σ_j R_{ij}
        C = Σ_i ρ_i C_i where ρ_i = Σ_j R_{ij} / Σ_{ij} R_{ij}
        
        Args:
            R: Importance matrix of shape (d, m).
            
        Returns:
            C: Completeness score in [0, 1].
        """
        d, m = R.shape
        
        # Normalize importance for each factor (row-wise)
        row_sums = R.sum(axis=1) + 1e-10
        P = R / row_sums[:, np.newaxis]  # P[i, j] = R[i,j] / Σ_j R[i,j]
        
        # Compute entropy for each factor
        H_i = np.zeros(d)
        for i in range(d):
            probs = P[i, :]
            # Entropy: -Σ p log(p), handling zeros
            mask = probs > 1e-10
            if mask.sum() > 0:
                H_i[i] = -np.sum(probs[mask] * np.log(probs[mask]))
        
        # Normalize entropy by log(m) to get values in [0, 1]
        max_entropy = np.log(m) if m > 1 else 1.0
        H_i_normalized = H_i / max_entropy
        
        # Completeness per factor: C_i = 1 - H_i_normalized
        C_i = 1.0 - H_i_normalized
        
        # Weighted average by factor importance
        total_importance = R.sum()
        if total_importance < 1e-10:
            return 0.0
        
        rho_i = row_sums / total_importance
        C = np.sum(rho_i * C_i)
        
        return float(np.clip(C, 0.0, 1.0))
    
    def _compute_informativeness(
        self, Z: np.ndarray, Z_hat: np.ndarray
    ) -> float:
        """
        Compute informativeness score.
        
        Measures how well the codes can predict the factors.
        Uses R² score averaged across factors.
        
        Args:
            Z: Ground-truth factors of shape (n, d).
            Z_hat: Learned codes of shape (n, m).
            
        Returns:
            I: Informativeness score in [0, 1].
        """
        d = Z.shape[1]
        r2_scores = np.zeros(d)
        
        for i in range(d):
            if self.method == "lasso":
                model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=3,
                    random_state=42
                )
            
            # Predict factor Z[:, i] from codes Z_hat
            model.fit(Z_hat, Z[:, i])
            r2 = model.score(Z_hat, Z[:, i])
            r2_scores[i] = max(0.0, r2)  # Clip negative R² to 0
        
        return float(np.mean(r2_scores))
