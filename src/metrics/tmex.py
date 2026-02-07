"""
T-MEX (Testing for Measurement Exchangeability) metric.

Based on "A Measurement Perspective of Causal Representation Learning".
Original implementation: https://github.com/CausalLearningAI/tmex

T-MEX tests whether there is a one-to-one mapping between ground-truth factors
and learned representations by performing conditional independence tests using
the Projected Covariance Measure (PCM) from the pycomets package.

The core PCM implementation is copied from pycomets
(https://github.com/shimenghuang/pycomets, GNU GENERAL PUBLIC LICENSE - Shimeng Huang).
"""

import copy
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment, root_scalar
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from typing import Optional, Literal

from .base import BaseMetric, MetricResult


# ---------------------------------------------------------------------------
# Regression methods copied from pycomets/regression.py
# ---------------------------------------------------------------------------

class RegressionMethod:
    def __init__(self, model):
        self.model = model
        self.model_fitted = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def residuals(self):
        raise NotImplementedError


class LM(RegressionMethod, BaseEstimator):
    def __init__(self, **kwargs):
        model = LinearRegression(**kwargs)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class RF(RegressionMethod, BaseEstimator):
    def __init__(self, **kwargs):
        self.resid_type = "vanilla"
        model = RandomForestRegressor(**kwargs)
        super().__init__(model)

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class KRR(RegressionMethod, BaseEstimator):
    def __init__(self, **kwargs):
        self.resid_type = "vanilla"
        model = KernelRidge(**kwargs)
        super().__init__(model)

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


# ---------------------------------------------------------------------------
# Utility functions copied from pycomets/utils.py
# ---------------------------------------------------------------------------

def _safe_squeeze(arr, axis=1):
    """Squeeze out axis if it is 1."""
    if arr.ndim == 1:
        return arr
    if axis >= arr.ndim:
        raise IndexError(f"axis {axis} exceeds the dimension of arr {arr.ndim}")
    if arr.shape[axis] == 1:
        return np.squeeze(arr.copy(), axis=axis)
    else:
        return arr


def _data_check(Y, X, Z):
    """Check data dimensions and reshape if needed."""
    Y_new, X_new, Z_new = Y.copy(), X.copy(), Z.copy()
    if Y.ndim > 1:
        Y_new = _safe_squeeze(Y_new, axis=1)
    if X.ndim > 1:
        X_new = _safe_squeeze(X_new, axis=1)
    if Z.ndim == 1:
        Z_new = Z_new[:, np.newaxis]
    return Y_new, X_new, Z_new


def _split_sample(Y, X, Z, test_split=0.5, rng=np.random.default_rng()):
    nn = Y.shape[0]
    idx_tr = rng.choice(
        np.arange(nn), replace=False, size=int(np.ceil(nn * (1 - test_split)))
    )
    idx_te = np.setdiff1d(np.arange(nn), idx_tr, assume_unique=True)
    return Y[idx_tr], X[idx_tr], Z[idx_tr], Y[idx_te], X[idx_te], Z[idx_te]


# ---------------------------------------------------------------------------
# PCM test copied from pycomets/pcm.py (MIT License)
# ---------------------------------------------------------------------------

def _pcm_test(
    Y,
    X,
    Z,
    reg_yonxz: RegressionMethod = RF(),
    reg_yonz: RegressionMethod = RF(),
    reg_yhatonz: RegressionMethod = RF(),
    reg_vonxz: RegressionMethod = RF(),
    reg_ronz: RegressionMethod = RF(),
    estimate_variance=True,
    test_split=0.5,
    max_exp=5,
    rng=np.random.default_rng(),
):
    """Computation of the PCM test with data splitting."""

    # sample splitting
    Y, X, Z = _data_check(Y, X, Z)
    if Y.ndim > 1:
        raise ValueError("PCM does not support multi-dimensional Y.")
    Ytr, Xtr, Ztr, Yte, Xte, Zte = _split_sample(Y, X, Z, test_split, rng)

    # regression on the training data (direction estimate)
    XZtr = np.column_stack([Xtr, Ztr])
    reg_yonxz.fit(Y=Ytr, X=XZtr)
    yhat = reg_yonxz.predict(X=XZtr)
    reg_yhatonz.fit(Y=yhat, X=Ztr)
    rho = np.mean((Ytr - reg_yhatonz.predict(X=Ztr)) * yhat)

    def hhat(X, Z):
        htilde = reg_yonxz.predict(np.column_stack([X, Z])) - reg_yhatonz.predict(Z)
        return np.sign(rho) * htilde

    # estimate variance
    if estimate_variance:
        sqr = (Ytr - yhat) ** 2
        reg_vonxz.fit(Y=sqr, X=XZtr)

        def a(c):
            den = np.column_stack(
                [reg_vonxz.predict(XZtr), np.repeat(0, XZtr.shape[0])]
            )
            return np.mean(sqr / (np.max(den, axis=1) + c)) - 1

        if a(0) < 0:
            chat = 0
        else:
            lwr, upr = 0, 10
            counter = 0
            while np.sign(a(lwr)) * np.sign(a(upr)) == 1:
                upr += 5
                counter += 1
                if counter > max_exp:
                    raise ValueError(
                        "Cannot compute variance estimate, try rerunning "
                        "with `estimate_variance=False`."
                    )
            chat = root_scalar(a, method="brentq", bracket=[lwr, upr]).root

        def vhat(X, Z):
            XZ = np.column_stack([X, Z])
            vtemp = np.max(
                np.column_stack(
                    [reg_vonxz.predict(XZ), np.repeat(0, XZ.shape[0])]
                ),
                axis=1,
            )
            return vtemp + chat
    else:

        def vhat(X, Z):
            return 1

    # regression on the test data
    def fhat(X, Z):
        return hhat(X, Z) / vhat(X, Z)

    fhats = fhat(Xte, Zte)
    reg_ronz.fit(Y=fhats, X=Zte)
    reg_yonz.fit(Y=Yte, X=Zte)

    # test
    rY = Yte - reg_yonz.predict(X=Zte)
    rT = fhats - reg_ronz.predict(X=Zte)
    L = rY * rT
    stat = (
        np.sqrt(Yte.shape[0])
        * np.mean(L)
        / np.sqrt(np.mean(L**2) - np.mean(L) ** 2)
    )
    if np.isnan(stat):
        stat = -np.inf
    pval = 1 - norm().cdf(stat)

    return pval, stat, rY, rT


class PCM:
    """
    Projected covariance measure test for conditional mean independence.
    Copied from pycomets.
    """

    def __init__(self):
        self.pval = None
        self.stat = None
        self.pvals = None
        self.stats = None

    def test(
        self,
        Y,
        X,
        Z,
        rep=1,
        reg_yonxz: RegressionMethod = RF(),
        reg_yonz: RegressionMethod = RF(),
        reg_yhatonz: RegressionMethod = RF(),
        reg_vonxz: RegressionMethod = RF(),
        reg_ronz: RegressionMethod = RF(),
        estimate_variance=True,
        test_split=0.5,
        max_exp=5,
        rng=np.random.default_rng(),
        show_summary=True,
    ):

        self.pvals = np.empty(rep)
        self.stats = np.empty(rep)
        n_test = int(np.floor(Y.shape[0] * test_split))
        self.rY = np.empty((n_test, rep))
        self.rT = np.empty((n_test, rep))
        for ii in range(rep):
            self.pvals[ii], self.stats[ii], self.rY[:, ii], self.rT[:, ii] = (
                _pcm_test(
                    Y,
                    X,
                    Z,
                    reg_yonxz,
                    reg_yonz,
                    reg_yhatonz,
                    reg_vonxz,
                    reg_ronz,
                    estimate_variance,
                    test_split,
                    max_exp,
                    rng,
                )
            )
        self.stat = np.mean(self.stats)
        self.pval = 1 - norm().cdf(self.stat)
        if show_summary:
            self.summary()

    def summary(self, digits=3):
        print("\tProjected covariance measure test")
        print(f"Z = {self.stat:.{digits}f}, p-value = {self.pval:.{digits}f}")
        print(
            "alternative hypothesis: true E[Y | X, Z] is not equal to E[Y | Z]"
        )


# ---------------------------------------------------------------------------
# T-MEX computation — follows comp_tmex() from small_example.py
# ---------------------------------------------------------------------------

def _comp_tmex(Z, Z_hat, fun_reg=None, alpha=0.05, rep=9, seed=None):
    """
    Compute the T-MEX correspondence matrix and error count.

    Follows the original ``comp_tmex`` from ``small_example.py`` as closely
    as possible.  For every pair (hat_z_i, z_j) we run a PCM conditional
    independence test of hat_z_i _|_ z_j | Z_{-j} and record whether we
    reject (p < alpha => 1) or not (0).

    Args:
        Z: Ground-truth factors, shape (n, d).
        Z_hat: Learned representations, shape (n, m).
        fun_reg: Regression method instance (LM(), RF(), or KRR()).
                 Defaults to LM() following the original.
        alpha: Significance level for independence tests.
        rep: Number of PCM repetitions (default 9, as in original).
        seed: Optional random seed.

    Returns:
        n_errors: Number of mismatches between W_hat and the identity
                  (lower is better - 0 means perfect identification).
        W_hat: Estimated correspondence matrix of shape (m, d).
    """
    if fun_reg is None:
        fun_reg = LM()

    d = Z.shape[1]
    m = Z_hat.shape[1]

    W = np.eye(m, d)  # expected / perfect correspondence
    W_hat = np.zeros((m, d))

    rng = np.random.default_rng(seed)

    # go through each block (learned representation)
    for ii in range(m):
        # go through each latent (ground-truth factor)
        for jj in range(d):
            pcm = PCM()
            pcm.test(
                reg_yonxz=copy.deepcopy(fun_reg),
                reg_ronz=copy.deepcopy(fun_reg),
                reg_vonxz=copy.deepcopy(fun_reg),
                reg_yhatonz=copy.deepcopy(fun_reg),
                reg_yonz=copy.deepcopy(fun_reg),
                X=Z[:, jj:jj+1],
                Y=Z_hat[:, ii:ii+1],
                Z=np.delete(Z, jj, axis=1),
                estimate_variance=False,
                rep=rep,
                rng=rng,
                show_summary=False,
            )
            W_hat[ii, jj] = pcm.pval < alpha

    n_errors = int(np.sum(W_hat != W))
    return n_errors, W_hat


def _compute_tmex_score(W_hat, aggregation="alignment"):
    """
    Derive a scalar score in [0, 1] from the correspondence matrix.

    Args:
        W_hat: Binary correspondence matrix, shape (m, d).
        aggregation:
            ``"alignment"`` - Hungarian matching, fraction of matched 1s.
            ``"error_count"`` - 1 - #errors / (m * d).

    Returns:
        Score in [0, 1], higher is better.
    """
    m, d = W_hat.shape

    if aggregation == "alignment":
        row_ind, col_ind = linear_sum_assignment(-W_hat)
        matches = sum(W_hat[i, j] for i, j in zip(row_ind, col_ind))
        return float(np.clip(matches / min(m, d), 0, 1))

    elif aggregation == "error_count":
        W_expected = np.eye(m, d)
        errors = int(np.sum(W_hat != W_expected))
        return float(np.clip(1 - errors / (m * d), 0, 1))

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# ---------------------------------------------------------------------------
# BaseMetric wrapper
# ---------------------------------------------------------------------------

class TMEXMetric(BaseMetric):
    """
    T-MEX (Testing for Measurement Exchangeability) metric.

    Wraps ``comp_tmex`` (PCM-based conditional independence testing)
    into the ``BaseMetric`` interface.

    Args:
        alpha: Significance level for independence tests.
        aggregation: How to turn the correspondence matrix into a scalar
            (``"alignment"`` or ``"error_count"``).
        rep: Number of PCM repetitions per test.
        regression_method: ``"lm"``, ``"rf"``, or ``"krr"``.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        aggregation: Literal["alignment", "error_count"] = "alignment",
        rep: int = 9,
        regression_method: str = "lm",
        seed: Optional[int] = None,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.aggregation = aggregation
        self.rep = rep
        self.regression_method = regression_method
        self.seed = seed

    @property
    def required_min_samples(self) -> int:
        return 50

    def _compute_impl(self, Z: np.ndarray, Z_hat: np.ndarray) -> MetricResult:
        # Build regression method
        if self.regression_method == "lm":
            fun_reg = LM()
        elif self.regression_method == "rf":
            fun_reg = RF()
        elif self.regression_method == "krr":
            fun_reg = KRR()
        else:
            raise ValueError(
                f"Unknown regression_method: {self.regression_method}"
            )

        n_errors, W_hat = _comp_tmex(
            Z, Z_hat,
            fun_reg=fun_reg,
            alpha=self.alpha,
            rep=self.rep,
            seed=self.seed,
        )

        score = _compute_tmex_score(W_hat, aggregation=self.aggregation)

        return MetricResult(
            primary_score=score,
            metadata={
                "correspondence_matrix": W_hat.tolist(),
                "n_errors": n_errors,
                "alpha": self.alpha,
                "aggregation": self.aggregation,
                "rep": self.rep,
                "regression_method": self.regression_method,
            },
        )
