"""Rectangular Sinkhorn matching (PyTorch, GPU-compatible).

Contains:
- sinkhorn_rectangular_plan_pt (entropic OT with dustbin row/column)
- sinkhorn_soft_mcc_pt (soft MCC from train/test correlation matrices)
"""

from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


def sinkhorn_rectangular_plan_pt(
    affinity: "torch.Tensor",
    *,
    temperature: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> Tuple["torch.Tensor", float, float, int]:
    """Entropic relaxation of rectangular one-to-one matching.

    A single dustbin row/column absorbs the dimension imbalance.  The
    returned plan has probability-normalized mass; multiply its correlation
    term by ``legacy_scale = max(d,m)/min(d,m)`` to recover standard MCC
    scale.

    Returns (real_plan, legacy_scale, marginal_residual, iterations).
    """
    if torch is None:
        raise ImportError("PyTorch is required for Sinkhorn")
    if affinity.ndim != 2 or affinity.numel() == 0:
        raise ValueError("affinity must be a non-empty 2-D tensor")
    if not torch.isfinite(affinity).all():
        raise ValueError("affinity must contain only finite values")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0:
        raise ValueError("tol must be positive")

    s = affinity.to(dtype=torch.float32)
    d, m = s.shape
    q = max(d, m)
    k = min(d, m)
    device, dtype = s.device, s.dtype

    if d < m:
        aug = torch.cat([s, torch.zeros((1, m), device=device, dtype=dtype)], dim=0)
        a = torch.cat(
            [
                torch.full((d,), 1.0 / q, device=device, dtype=dtype),
                torch.tensor([(m - d) / q], device=device, dtype=dtype),
            ]
        )
        b = torch.full((m,), 1.0 / q, device=device, dtype=dtype)
    elif d > m:
        aug = torch.cat([s, torch.zeros((d, 1), device=device, dtype=dtype)], dim=1)
        a = torch.full((d,), 1.0 / q, device=device, dtype=dtype)
        b = torch.cat(
            [
                torch.full((m,), 1.0 / q, device=device, dtype=dtype),
                torch.tensor([(d - m) / q], device=device, dtype=dtype),
            ]
        )
    else:
        aug = s
        a = torch.full((d,), 1.0 / q, device=device, dtype=dtype)
        b = torch.full((m,), 1.0 / q, device=device, dtype=dtype)

    log_p = aug / temperature
    log_a = torch.log(a)
    log_b = torch.log(b)

    for it in range(max_iter):
        log_p = log_p + log_a[:, None] - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p + log_b[None, :] - torch.logsumexp(log_p, dim=0, keepdim=True)
        if (it + 1) % 10 == 0 or it + 1 == max_iter:
            p = torch.exp(log_p)
            row_err = torch.max(torch.abs(p.sum(dim=1) - a))
            col_err = torch.max(torch.abs(p.sum(dim=0) - b))
            residual = float(torch.maximum(row_err, col_err).detach().cpu())
            if residual <= tol:
                break

    p = torch.exp(log_p)
    return p[:d, :m], q / k, residual, it + 1


def sinkhorn_soft_mcc_pt(
    r_train: "torch.Tensor",
    r_test: Optional["torch.Tensor"] = None,
    *,
    temperature: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-5,
    coverage_aware: bool = False,
) -> Tuple["torch.Tensor", dict]:
    """Soft Sinkhorn MCC from train/test correlation matrices.

    With r_test provided, the plan and signs are learned from r_train and
    the signed score is evaluated on r_test (not abs(r_test)).
    """
    if torch is None:
        raise ImportError("PyTorch is required for Sinkhorn")
    if r_test is not None and r_test.shape != r_train.shape:
        raise ValueError("r_train and r_test must have the same shape")

    plan, legacy_scale, residual, iterations = sinkhorn_rectangular_plan_pt(
        torch.abs(r_train),
        temperature=temperature,
        max_iter=max_iter,
        tol=tol,
    )
    d, m = r_train.shape
    q = max(d, m)
    scale = (q / d) if coverage_aware else legacy_scale

    if r_test is None:
        score = scale * torch.sum(plan * torch.abs(r_train))
    else:
        score = scale * torch.sum(plan * torch.sign(r_train) * r_test)

    return score, {
        "temperature": temperature,
        "iterations": iterations,
        "marginal_residual": residual,
        "legacy_scale": legacy_scale,
        "coverage_aware": coverage_aware,
    }
