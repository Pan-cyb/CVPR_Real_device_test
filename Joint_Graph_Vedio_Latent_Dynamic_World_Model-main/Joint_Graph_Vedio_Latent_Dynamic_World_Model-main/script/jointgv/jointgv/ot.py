"""Small, dependency-free differentiable Sinkhorn utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_cost(x: Tensor, y: Tensor) -> Tensor:
    """Squared Euclidean costs for [B, N, D] and [B, M, D]."""
    # Do not implement this as ``torch.cdist(...).square()``. Sinkhorn
    # divergence evaluates x against itself, so its diagonal contains exact
    # zero distances. Some CUDA/PyTorch combinations can produce a finite
    # forward value but a non-finite gradient through cdist's square root at
    # those zero-distance entries. This algebraically equivalent expression
    # never takes a square root and has a well-defined zero-distance gradient.
    x_norm = x.square().sum(dim=-1, keepdim=True)
    y_norm = y.square().sum(dim=-1).unsqueeze(-2)
    cost = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return cost.clamp_min(0.0)


def sinkhorn_plan(x: Tensor, y: Tensor, epsilon: float = 0.1, iters: int = 30) -> Tensor:
    """Balanced entropy-regularized OT plan with uniform empirical weights."""
    cost = pairwise_cost(x, y)
    batch, n, m = cost.shape
    log_a = cost.new_full((batch, n), -torch.log(cost.new_tensor(float(n))))
    log_b = cost.new_full((batch, m), -torch.log(cost.new_tensor(float(m))))
    kernel = -cost / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(kernel + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(kernel + log_u[:, :, None], dim=1)
    return torch.exp(kernel + log_u[:, :, None] + log_v[:, None, :])


def sinkhorn_ot(x: Tensor, y: Tensor, epsilon: float = 0.1, iters: int = 30) -> Tensor:
    plan = sinkhorn_plan(x, y, epsilon, iters)
    return (plan * pairwise_cost(x, y)).sum(dim=(1, 2))


def sinkhorn_divergence(
    x: Tensor, y: Tensor, epsilon: float = 0.1, iters: int = 30
) -> Tensor:
    divergence, _, _, _ = sinkhorn_divergence_components(
        x, y, epsilon, iters
    )
    return divergence


def sinkhorn_divergence_components(
    x: Tensor, y: Tensor, epsilon: float = 0.1, iters: int = 30
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return divergence and its cross/self transport components.

    The three component values are diagnostics for distinguishing genuine
    cross-modal alignment from collapsed point clouds.  The first return
    value is exactly the loss historically returned by
    :func:`sinkhorn_divergence`.
    """
    xy = sinkhorn_ot(x, y, epsilon, iters)
    xx = sinkhorn_ot(x, x, epsilon, iters)
    yy = sinkhorn_ot(y, y, epsilon, iters)
    # The exact divergence is non-negative; clamp tiny negative floating-point
    # residuals so they cannot reward projection-scale pathologies.
    divergence = (xy - 0.5 * xx - 0.5 * yy).clamp_min(0.0)
    return divergence.mean(), xy.mean(), xx.mean(), yy.mean()


def transport_values(query: Tensor, source: Tensor, values: Tensor, epsilon: float, iters: int) -> Tensor:
    """Transport source values to query positions, returning [B, N, Dv]."""
    plan = sinkhorn_plan(query, source, epsilon, iters)
    row_mass = plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.bmm(plan / row_mass, values)
