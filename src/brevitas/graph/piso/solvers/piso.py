# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PiSO interval-sweep solvers (paper Algorithm 1).

Given the Hessians H (and optional G) and the quantization grid, each solver
computes the reconstruction-optimal weight quantization scale by sweeping the
intervals on which the round-to-nearest assignment q(w; s) is constant and
evaluating the closed-form per-interval minimizer.

The _piso_sweep_* functions implement the same interval sweep: sort transition
scales, walk intervals from large to small |s| maintaining alpha = q^T G w and
beta = q^T H q (code: wHq, qHq), take the closed-form s* = alpha / beta per
interval, keep the global best. They differ only in the structure exploited:

  _piso_sweep_per_channel_diag          diagonal H, per channel  (O(D|G|))
  _piso_sweep_per_group_diag            diagonal H, groups independent

The dense-H sweeps (per channel, per group, and the sequential group variants)
are not implemented.

The per-interval update (how wHq/qHq change) differs per variant and stays
inline. The shared grid-delta and closed-form/optimality helpers live in
.common. The chunked reformulation lives in .chunked_piso.

Reference: "Optimal Post-Training Quantization Scales and Where to Find Them".
"""

from typing import Optional

import torch

from brevitas.graph.piso.solvers.common import _piso_grid_deltas
from brevitas.graph.piso.solvers.common import _piso_select_optimal
from brevitas.graph.piso.solvers.common import _piso_select_optimal_straddle
from brevitas.graph.piso.solvers.common import SCALE_SOLVER_REGISTRY
from brevitas.graph.piso.solvers.common import ScaleSolverFamily


def _piso_sweep_per_channel_diag(
        w_b: torch.Tensor,
        H_diag: torch.Tensor,
        unscaled_grid: torch.Tensor,
        G_diag: Optional[torch.Tensor] = None,
        eps: float = 1.e-6,
        allow_negative_s: bool = False) -> torch.Tensor:
    """PiSO sweep for per-channel quantization with a diagonal Hessian (paper section 3.1).

    Fast path when H is forced diagonal (debug) or the data-free regime is used
    (H = I). No cross-weight interactions, so the auxiliary vector h = H q is not
    needed and each interval update is O(1), giving O(D|grid|) per channel.

    Args:
        w_b: weights, shape [batch, D].
        H_diag: diagonal of H = X_tilde^T X_tilde, shape [D].
        G_diag: diagonal of G = X_tilde^T X for the cross-activation objective,
            shape [D]. If None, H_diag is used for both terms.
        unscaled_grid: sorted quantization grid with 0 included.
        eps: interval degeneracy threshold and clamping margin.
        allow_negative_s: whether to search negative scales (asymmetric grids).

    Returns:
        Optimal scale per channel, shape [batch, 1].
    """
    _, k_curr, k_next, delta_next_curr = _piso_grid_deltas(unscaled_grid)

    # batch size and dimension of weight vectors
    batch_size, D = w_b.shape
    # Batch indices for batched operations
    rows = torch.arange(batch_size, device=w_b.device)

    # Compute scale values where the vector q changes
    transition_scales = 2 * w_b.unsqueeze(1) / (unscaled_grid[:-1] +
                                                unscaled_grid[1:]).unsqueeze(1).unsqueeze(0)

    # Sort the transition scales to have the intervals where q is constant
    sorted_transition_scales, scales_sorting_indices = torch.sort(transition_scales.reshape(batch_size, -1), dim=1)

    # Intialization of values for the loop
    # Hw doesn't change during the loop
    if G_diag is not None:
        Hw = G_diag * w_b
    else:
        Hw = H_diag * w_b
    # wHq is originally all 0s as Hq is originally 0s
    wHq = torch.zeros((batch_size,), dtype=w_b.dtype, device=w_b.device)
    # qHq is originally all 0s as Hq is originally 0s
    qHq = torch.zeros((batch_size,), dtype=w_b.dtype, device=w_b.device)
    # There's no current optimal scale until the end of the first loop iteration
    optimal_scale = torch.empty((batch_size,), device=w_b.device)
    # Current optimal error is infinity as there's no current optimal scale
    optimal_error = torch.full((batch_size,), torch.inf, device=w_b.device)

    # Ensure qs are updated only once when there is a sign change in the scale
    mask_sign = torch.zeros((batch_size,), dtype=torch.bool, device=w_b.device)
    # Intialize values from positive to negative scale transition
    # This is the only transition where more than one entry of q changes at a time
    q_transition = torch.where(w_b[rows] < 0., unscaled_grid[-1],
                               unscaled_grid[0]).to(dtype=w_b.dtype)
    wHq_transition = torch.sum(Hw * q_transition, dim=1)
    Hq_transition = H_diag * q_transition
    qHq_transition = torch.sum(q_transition * Hq_transition, dim=1)

    # Loop over the intervals with fixed q compute the closed solution for
    # the best scale in the interval and if the error of the best scale in
    # the interval is lower than the optimal error, update the optimal scale
    for j in reversed(range(1, transition_scales.numel() // batch_size)):
        # i is the changed entry of the vector q in this iteration
        # k allows to retrieve the ammount of change in that entry
        k, i = scales_sorting_indices[:, j] // D, scales_sorting_indices[:, j] % D

        # Track scale sign changes
        # negative scales
        mask = sorted_transition_scales[:, j] < 0
        # sign changes happen if scale is negative and wasnt negative before
        mask_sign_changes = mask & ~mask_sign
        # scale intervals are scanned from right to left.
        # Once negative intervals are entered all new explored scales are negative
        mask_sign |= mask

        # Handle the special case of going from positive to negative scales
        if torch.any(mask_sign_changes):
            wHq[mask_sign_changes] = wHq_transition[mask_sign_changes]
            qHq[mask_sign_changes] = qHq_transition[mask_sign_changes]

        # get the change in the single entry of q that got updated
        delta = delta_next_curr[k] * (1. - 2. * mask.to(dtype=w_b.dtype))

        # get the bounds of the current scale interval
        upper_limit = sorted_transition_scales[:, j]
        lower_limit = sorted_transition_scales[:, j - 1]

        # iterative updates exploiting that only a single entry of
        # q changes, by a known quantity delta. Diagonal H: no Hq to maintain,
        # but the quadratic term needs the entry's *current* grid value: k_curr on
        # the positive branch, and its mirror k_next once s has gone negative (the
        # sign transition swaps which side of the grid the entry is walking from).
        k_base = torch.where(mask, k_next[k], k_curr[k])
        wHq += delta * Hw[rows, i]
        qHq += H_diag[i] * (2. * k_base * delta + delta ** 2)

        optimal_scale, optimal_error, should_break, _ = _piso_select_optimal(
            wHq, qHq, lower_limit, upper_limit, mask, optimal_scale, optimal_error, eps,
            allow_negative_s)

        optimal_scale, optimal_error = _piso_select_optimal_straddle(
            wHq_transition, qHq_transition, lower_limit, upper_limit, mask, optimal_scale,
            optimal_error, eps, allow_negative_s)

        if should_break:
            break

    return optimal_scale.unsqueeze(1)


def _piso_sweep_per_group_diag(
        w_b: torch.Tensor,
        H_diag: torch.Tensor,
        unscaled_grid: torch.Tensor,
        G_diag: Optional[torch.Tensor] = None,
        eps: float = 1.e-6,
        allow_negative_s: bool = False) -> torch.Tensor:
    """PiSO sweep for group-wise quantization, independent groups with a diagonal Hessian (paper section 3.1, section 3.2).

    Combines the independent-groups approximation (section 3.2) with the diagonal-H fast
    path (section 3.1): groups are treated independently and H is diagonal within each
    group, so no cross-weight or cross-group interactions are modelled. O(G|grid|)
    per group. Used for the data-free regime and the forced-diagonal debug path.

    Args:
        w_b: weights, shape [batch, n_groups, group_size].
        H_diag: diagonal of H = X_tilde^T X_tilde, shape [D] (D = n_groups * group_size).
        G_diag: diagonal of G = X_tilde^T X, shape [D].
            If None, H_diag is used for both terms.
        unscaled_grid: sorted quantization grid with 0 included.
        eps: interval degeneracy threshold and clamping margin.
        allow_negative_s: whether to search negative scales (asymmetric grids).

    Returns:
        Optimal scale per group, shape [batch, n_groups, 1].
    """
    _, k_curr, k_next, delta_next_curr = _piso_grid_deltas(unscaled_grid)

    # batch size, groups and group dimension of weights
    batch_size, n_groups, group_size = w_b.shape

    g_idx = torch.arange(n_groups, device=w_b.device).unsqueeze(0).expand(batch_size, -1)

    # Compute scale values where the vector q changes
    transition_scales = 2 * w_b.unsqueeze(1) / (unscaled_grid[:-1] +
                                                unscaled_grid[1:]).unsqueeze(1).unsqueeze(0).view(
                                                    1, unscaled_grid.shape[0] - 1, 1, 1)
    # Sort the transition scales to have the intervals where q is constant

    sorted_transition_scales, scales_sorting_indices = torch.sort(transition_scales.permute(0, 2, 1, 3).reshape(batch_size, n_groups, -1), dim=-1)

    if G_diag is not None:
        Hw = (G_diag * w_b.reshape(batch_size, n_groups * group_size)).reshape(
            batch_size, n_groups, group_size)
    else:
        Hw = (H_diag * w_b.reshape(batch_size, n_groups * group_size)).reshape(
            batch_size, n_groups, group_size)

    wHq = torch.zeros((batch_size, n_groups), dtype=w_b.dtype, device=w_b.device)
    qHq = torch.zeros((batch_size, n_groups), dtype=w_b.dtype, device=w_b.device)
    optimal_scale = torch.empty((batch_size, n_groups), device=w_b.device)
    # Current optimal error is infinity as there's no current optimal scale
    optimal_error = torch.full((batch_size, n_groups), torch.inf, device=w_b.device)

    mask_sign = torch.zeros((batch_size, n_groups), dtype=torch.bool, device=w_b.device)
    q_transition = torch.where(w_b < 0., unscaled_grid[-1], unscaled_grid[0]).to(dtype=w_b.dtype)
    wHq_transition = torch.sum(Hw * q_transition, dim=2)
    Hq_transition = (H_diag * q_transition.reshape(batch_size, n_groups * group_size)).reshape(
        batch_size, n_groups, group_size)
    qHq_transition = torch.sum(q_transition * Hq_transition, dim=2)

    for j in reversed(range(1, transition_scales.numel() // (batch_size * n_groups))):
        k, i = scales_sorting_indices[:,:,j] // group_size, scales_sorting_indices[:,:,j] % group_size
        mask = sorted_transition_scales[:, :, j] < 0
        mask_sign_changes = mask & ~mask_sign
        mask_sign |= mask

        if torch.any(mask_sign_changes):
            wHq[mask_sign_changes] = wHq_transition[mask_sign_changes]
            qHq[mask_sign_changes] = qHq_transition[mask_sign_changes]

        delta = delta_next_curr[k] * (1. - 2. * mask.to(dtype=w_b.dtype))
        upper_limit = sorted_transition_scales[:, :, j]
        lower_limit = sorted_transition_scales[:, :, j - 1]

        # Diagonal H: the quadratic term needs the entry's current grid value —
        # k_curr on the positive branch, its mirror k_next once s is negative.
        k_base = torch.where(mask, k_next[k], k_curr[k])
        wHq += delta * torch.gather(Hw, dim=2, index=i.unsqueeze(-1)).squeeze(-1)
        qHq += H_diag[g_idx * group_size + i] * (2. * k_base * delta + delta ** 2)

        optimal_scale, optimal_error, should_break, _ = _piso_select_optimal(
            wHq, qHq, lower_limit, upper_limit, mask, optimal_scale, optimal_error, eps,
            allow_negative_s)

        optimal_scale, optimal_error = _piso_select_optimal_straddle(
            wHq_transition, qHq_transition, lower_limit, upper_limit, mask, optimal_scale,
            optimal_error, eps, allow_negative_s)

        if should_break:
            break

    return optimal_scale.unsqueeze(-1)


@SCALE_SOLVER_REGISTRY.register(names='piso')
class PiSOSolverFamily(ScaleSolverFamily):
    """Paper Algorithm 1, diagonal H; symmetric and asymmetric grids.

    The dense-H slots (per_channel, per_group, groups_sequential,
    single_group_sequential) are not implemented, so the base stubs raise
    NotImplementedError.
    """

    _per_channel_diag = staticmethod(_piso_sweep_per_channel_diag)
    _per_group_diag = staticmethod(_piso_sweep_per_group_diag)
