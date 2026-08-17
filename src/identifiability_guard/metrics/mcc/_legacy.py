"""Legacy icebeem MCC functions (NumPy + PyTorch).

Adapted from https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py

Contains:
- RDC (Randomized Dependence Coefficient)
- Auction linear assignment (PyTorch, GPU)
- PyTorch correlation helpers (rankdata, cov, corrcoef, spearmanr)
- cross_correlation_pt (fast GPU d×m correlation via cuBLAS)
- mean_corr_coef_np / mean_corr_coef_pt / mean_corr_coef (dispatcher)
- mean_corr_coef_out_of_sample (deprecated)
"""

import warnings

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# ============================================================================
# RDC (Randomized Dependence Coefficient)
# ============================================================================


def rdc(x, y, k=20, s=0.5, nonlinearity="sin", rng=None):
    """
    Python implementation of the Randomized Dependence Coefficient (RDC) [1] algorithm
    the RDC is a measure of correlation between two (scalar) random variables x and y
    that is invariant to permutation, scaling, and most importantly nonlinear scaling

    Parameters:
        x: numpy array of shape (n,)
        y: numpy array of shape (n,)
        k: number of random projections in RDC
        s: covariance of the Gaussian dist used for sampling the random weights
        nonlinearity: nonlinear feature map used to transform the random projections
        rng: numpy random Generator for reproducibility

    Return:
        rdc_cc: flaot in [0,1] --- the RDC correlation coefficient

    References:
    [1] https://papers.nips.cc/paper/2013/file/aab3238922bcc25a6f606eb525ffdc56-Paper.pdf

    """
    if rng is None:
        rng = np.random.default_rng()
    cx = copula_projection(x, k, s, nonlinearity, rng=rng)
    cy = copula_projection(y, k, s, nonlinearity, rng=rng)
    rdc_cc = largest_cancorr(cx, cy)
    return rdc_cc


def copula_projection(x, k=20, s=0.5, nonlinearity="sin", rng=None):
    n = x.shape[0]
    k = min(k, n)
    p = rank_array(x) / n
    pt = np.vstack([p, np.ones(n)]).T
    if rng is None:
        rng = np.random.default_rng()
    wt = rng.normal(0, s, size=(pt.shape[1], k))
    if nonlinearity == "sin":
        phix = np.sin(pt.dot(wt))
    elif nonlinearity == "cos":
        phix = np.cos(pt.dot(wt))
    else:
        raise ValueError(f"{nonlinearity} not supported")
    return np.hstack([phix, np.ones((n, 1))])


def make_diag(el, nrows, ncols):
    return el * np.eye(nrows, ncols)


def largest_cancorr(x, y):
    """Return the largest canonical correlation between matrices x and y."""
    n = x.shape[0]
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    qx, _ = scipy.linalg.qr(x, mode="full")
    qy, _ = scipy.linalg.qr(y, mode="full")
    dx = np.linalg.matrix_rank(x)
    dy = np.linalg.matrix_rank(y)
    qxy = qx.T.dot(qy.dot(make_diag(1, n, dy)))[:dx]
    _, s, _ = scipy.linalg.svd(qxy, lapack_driver="gesvd")
    return s[0]


def rank_array(x):
    """Rank the elements of a vector."""
    tmp = x.argsort()
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(x))
    return ranks + 1


# ============================================================================
# Auction assignment (PyTorch, GPU)
# ============================================================================


def auction_linear_assignment(x, eps=None, reduce="sum", max_iter=10000):
    r"""
    Solve the linear sum assignment problem using the auction algorithm.
    Implementation in pytorch, GPU compatible.

    x_ij is the affinity between row (person) i and column (object) j, the
    algorithm aims to assign to each row i a column j_i such that the total benefit
    \sum_i x_{ij_i} is maximized.

    pytorch implementation, supports GPU.

    Algorithm adapted from http://web.mit.edu/dimitrib/www/Auction_Survey.pdf

    :param x: torch.Tensor
            The affinity (or benefit) matrix of size (n, n)
    :param eps: float, optional
            Bid size. Smaller values yield higher accuracy at the price of
            longer runtime.
    :param reduce: str, optional
            The reduction method to be applied to the score.
            If `sum`, sum the entries of cost matrix after assignment.
            If `mean`, compute the mean of the cost matrix after assignment.
            If `none`, return the vector (n,) of assigned column entry per row.
    :param max_iter: int, optional
            Maximum number of auction iterations before terminating.
    :return: (torch.Tensor, torch.Tensor, int)
            Tuple of (score after application of reduction method, assignment,
            number of steps in the auction algorithm).
    """
    eps = 1 / x.size(0) if eps is None else eps

    price = torch.zeros((1, x.size(1))).to(x.device)
    assignment = torch.zeros(x.size(0)).long().to(x.device) - 1
    bids = torch.zeros_like(x).to(x.device)

    n_iter = 0
    while (assignment == -1).any():
        n_iter += 1
        if n_iter > max_iter:
            warnings.warn(
                f"auction_linear_assignment: reached {max_iter} iterations "
                f"without full assignment; returning partial result."
            )
            break

        I = (assignment == -1).nonzero().squeeze(dim=1)
        value_I = x[I, :] - price
        top_value, top_idx = value_I.topk(2, dim=1)
        jI = top_idx[:, 0]
        vI, wI = top_value[:, 0], top_value[:, 1]
        gamma_I = vI - wI + eps
        bids_ = bids[I, :]
        bids_.zero_()
        bids_.scatter_(
            dim=1, index=jI.contiguous().view(-1, 1), src=gamma_I.view(-1, 1)
        )

        J = (bids_ > 0).sum(dim=0).nonzero().squeeze(dim=1)
        gamma_iJ, iJ = bids_[:, J].max(dim=0)
        iJ = I[iJ]
        price[:, J] += gamma_iJ
        mask = (assignment.view(-1, 1) == J.view(1, -1)).sum(dim=1).bool()
        assignment.masked_fill_(mask, -1)
        assignment[iJ] = J

    score = x.gather(dim=1, index=assignment.view(-1, 1)).squeeze()
    if reduce == "sum":
        score = torch.sum(score)
    elif reduce == "mean":
        score = torch.mean(score)
    elif reduce == "none":
        pass
    else:
        raise ValueError("not a valid reduction method: {}".format(reduce))

    return score, assignment, n_iter


# ============================================================================
# PyTorch correlation helpers
# ============================================================================


def rankdata_pt(b, tie_method="ordinal", dim=0):
    """PyTorch equivalent of scipy.stats.rankdata, GPU compatible."""
    if b.dim() > 2:
        raise ValueError("input has more than 2 dimensions")
    if b.dim() < 1:
        raise ValueError("input has less than 1 dimension")

    order = torch.argsort(b, dim=dim)

    if tie_method == "ordinal":
        ranks = torch.zeros_like(order)
        shape = [1] * b.dim()
        shape[dim] = b.size(dim)
        src = torch.arange(1, b.size(dim) + 1, device=b.device).view(shape).expand_as(b)
        ranks.scatter_(dim, order, src)
    else:
        if b.dim() != 1:
            raise NotImplementedError(
                "tie_method {} not supported for 2-D tensors".format(tie_method)
            )
        else:
            n = b.size(0)
            ranks = torch.empty(n).to(b.device)
            dupcount = 0
            total_tie_count = 0
            for i in range(n):
                inext = i + 1
                if i == n - 1 or b[order[i]] != b[order[inext]]:
                    if tie_method == "average":
                        tie_rank = inext - 0.5 * dupcount
                    elif tie_method == "min":
                        tie_rank = inext - dupcount
                    elif tie_method == "max":
                        tie_rank = inext
                    elif tie_method == "dense":
                        tie_rank = inext - dupcount - total_tie_count
                        total_tie_count += dupcount
                    else:
                        raise ValueError(
                            "not a valid tie_method: {}".format(tie_method)
                        )
                    for j in range(i - dupcount, inext):
                        ranks[order[j]] = tie_rank
                    dupcount = 0
                else:
                    dupcount += 1
    return ranks


def cov_pt(x, y=None, rowvar=False):
    """Estimate a covariance matrix in PyTorch, GPU compatible."""
    if y is not None:
        if x.size(0) != y.size(0):
            raise ValueError("x and y must contain the same number of samples")
    if x.dim() > 2:
        raise ValueError("x has more than 2 dimensions")
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()
    if y is not None:
        if y.dim() < 2:
            y = y.view(1, -1)
        if not rowvar and y.size(0) != 1:
            y = y.t()
        x = torch.cat((x, y), dim=0)

    n_samples = x.size(1)
    fact = 1.0 / max(n_samples - 1, 1)
    x = x - torch.mean(x, dim=1, keepdim=True)
    return fact * x.matmul(x.t()).squeeze()


def corrcoef_pt(x, y=None, rowvar=False):
    """Pearson correlation coefficients in PyTorch, GPU compatible."""
    c = cov_pt(x, y, rowvar)
    try:
        d = torch.diag(c)
    except RuntimeError:
        return c / c
    stddev = torch.sqrt(d.clamp(min=0))
    stddev = torch.where(stddev < 1e-12, torch.ones_like(stddev), stddev)
    c /= stddev[:, None]
    c /= stddev[None, :]
    return c


def spearmanr_pt(x, y=None, rowvar=False):
    """Spearman rank correlation in PyTorch, GPU compatible."""
    xr = rankdata_pt(x, dim=int(rowvar)).float()
    yr = None
    if y is not None:
        yr = rankdata_pt(y, dim=int(rowvar)).float()
    rs = corrcoef_pt(xr, yr, rowvar)
    return rs


def cross_correlation_pt(x, y, *, method="pearson", eps=1e-12):
    """GPU-accelerated d×m signed cross-correlation via cuBLAS matmul.

    Only computes the d×m cross block (not the full (d+m)×(d+m) covariance).
    Constant columns receive zero correlation.
    """
    if torch is None:
        raise ImportError("PyTorch is required")
    if method == "spearman":
        x = rankdata_pt(x, dim=0).float()
        y = rankdata_pt(y, dim=0).float()
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    xnorm = torch.linalg.norm(xc, dim=0)
    ynorm = torch.linalg.norm(yc, dim=0)
    denom = xnorm[:, None] * ynorm[None, :]
    numerator = xc.T @ yc
    safe_denom = torch.where(denom > eps, denom, torch.ones_like(denom))
    out = torch.where(denom > eps, numerator / safe_denom, torch.zeros_like(numerator))
    return torch.clamp(out, -1.0, 1.0)


# ============================================================================
# MCC dispatchers (NumPy / PyTorch)
# ============================================================================


def _mcc_assignment_scipy(cc, d, m):
    """Exact assignment via Jonker-Volgenant on CPU. Fast for all d."""
    rows, cols = linear_sum_assignment(cc.numpy(), maximize=True)
    return float(np.clip(cc[rows, cols].numpy().mean(), 0.0, 1.0))


def _mcc_assignment_auction(cc, d, m):
    """Auction assignment on GPU. Avoids device->host roundtrip."""
    k = min(d, m)
    if k == 1:
        return float(cc.max().clamp(0.0, 1.0))

    size = max(d, m)
    if d != m:
        cc_sq = torch.zeros((size, size), device=cc.device, dtype=cc.dtype)
        cc_sq[:d, :m] = cc
    else:
        cc_sq = cc

    _, assignment, _ = auction_linear_assignment(cc_sq, reduce="none")

    real_mask = assignment[:d] < m
    if not real_mask.any():
        return 0.0
    matched_cols = assignment[:d][real_mask]
    matched_rows = torch.arange(d, device=cc.device)[real_mask]
    score = cc[matched_rows, matched_cols].mean()
    return float(torch.clamp(score, 0.0, 1.0))


def mean_corr_coef_pt(x, y, method="pearson"):
    """PyTorch MCC: GPU correlation + assignment (auction on GPU, scipy on CPU)."""
    cc = cross_correlation_pt(x, y, method=method)
    d, m = cc.shape
    cc = torch.abs(cc)
    cc = torch.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)

    if cc.device.type != "cpu":
        return _mcc_assignment_auction(cc, d, m)
    return _mcc_assignment_scipy(cc, d, m)


def mean_corr_coef_np(x, y, method="pearson", rng=None):
    """NumPy implementation of legacy in-sample MCC."""
    d = x.shape[1]
    m = y.shape[1]
    if method == "pearson":
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == "spearman":
        cc = spearmanr(x, y)[0][:d, d:]
    elif method == "rdc":
        cc = np.zeros((d, m))
        for i in range(d):
            for j in range(m):
                cc[i, j] = rdc(x[:, i], y[:, j], rng=rng)
    else:
        raise ValueError("not a valid method: {}".format(method))
    cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
    cc = np.abs(cc)
    rows, cols = linear_sum_assignment(cc, maximize=True)
    score = cc[rows, cols].mean()
    return float(np.clip(score, 0.0, 1.0))


def mean_corr_coef(x, y, method="pearson", rng=None):
    """Dispatcher: selects NumPy or PyTorch MCC based on input type."""
    if type(x) != type(y):
        raise ValueError(f"inputs are of different types: ({type(x)}, {type(y)})")
    if isinstance(x, np.ndarray):
        return mean_corr_coef_np(x, y, method, rng=rng)
    elif HAS_TORCH and isinstance(x, torch.Tensor):
        return mean_corr_coef_pt(x, y, method)
    elif not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install torch\n"
            "Or convert your tensors to NumPy: x.cpu().numpy()"
        )
    else:
        raise ValueError(f"not a supported input type: {type(x)}")


def mean_corr_coef_out_of_sample(x, y, x_test, y_test, method="pearson", rng=None):
    """Legacy out-of-sample MCC using absolute test correlations.

    .. deprecated::
        Use ``mcc_train_test_np`` instead, which applies sign correction
        (giving zero expected score under the null) rather than taking
        absolute values on the test set.
    """
    warnings.warn(
        "mean_corr_coef_out_of_sample is deprecated; use mcc_train_test_np "
        "for sign-corrected cross-fitted evaluation",
        DeprecationWarning,
        stacklevel=2,
    )
    d = x.shape[1]
    m = y.shape[1]
    if method == "pearson":
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
        cc_test = np.corrcoef(x_test, y_test, rowvar=False)[:d, d:]
    elif method == "spearman":
        cc = spearmanr(x, y)[0][:d, d:]
        cc_test = spearmanr(x_test, y_test)[0][:d, d:]
    elif method == "rdc":
        cc = np.zeros((d, m))
        for i in range(d):
            for j in range(m):
                cc[i, j] = rdc(x[:, i], y[:, j], rng=rng)
        cc_test = np.zeros((d, m))
        for i in range(d):
            for j in range(m):
                cc_test[i, j] = rdc(x_test[:, i], y_test[:, j], rng=rng)
    else:
        raise ValueError("not a valid method: {}".format(method))
    cc = np.abs(cc)
    score = np.abs(cc_test)[linear_sum_assignment(-1 * cc)].mean()
    return float(np.clip(score, 0.0, 1.0))
