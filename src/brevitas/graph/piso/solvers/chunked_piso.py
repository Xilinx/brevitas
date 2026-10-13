# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Chunked PiSO (positive scales only).

An exact reformulation of the PiSO interval sweep (in .piso) that processes
CHUNK_SIZE transitions per chunk in parallel instead of one at a time. It
maintains a cumulative-delta vector V (with the invariant V = q at chunk
boundaries) and accumulates wHq = w^T G q and qHq = q^T H q via prefix sums
(torch.cumsum). Equivalent to the sequential sweep in exact arithmetic; float32
differences (~1e-7 for small D) come from summation-order non-commutativity.

These are the positive-scales-only variants: they drop all sign-change handling
and therefore only support symmetric grids (allow_negative_s=False). They raise a
ValueError for allow_negative_s=True; use the 'piso' family for asymmetric grids.

Only the diagonal slots are implemented. H is diagonal there, so (Hq)_i reduces
to the entry's current grid value (a grid-derived constant) and no Hq recovery is
needed; the per-transition setup and the per-chunk closed-form evaluation are
shared between them, and the group variant adds a groups batch dimension over the
flat per-channel math. The dense-H slots are not implemented.

Reference: "Equivalence of the Closed-Form and Chunked Interval-Sweep Scale
Optimisers" and "Unrolling the Interval-Sweep Recurrence".
"""

from typing import Optional
from typing import Tuple

import torch

from brevitas.graph.piso.solvers.common import _piso_grid_deltas
from brevitas.graph.piso.solvers.common import SCALE_SOLVER_REGISTRY
from brevitas.graph.piso.solvers.common import ScaleSolverFamily

# Number of transitions processed per chunk. Larger chunks trade more within-chunk
# O(C^2) work for fewer sequential iterations (kernel launches on GPU).
_CHUNKED_PISO_CHUNK_SIZE = 128


def _reject_negative_scales(allow_negative_s: bool) -> None:
    if allow_negative_s:
        raise ValueError(
            "chunked-piso solver only supports positive scales (symmetric grids); "
            "got allow_negative_s=True. Use the 'piso' solver for asymmetric grids.")


def _chunked_transition_setup(
    w_flat: torch.Tensor, unscaled_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Shared per-transition setup for every positive-only chunked sweep.

    Given flat weights w_flat (N, D) -- N being batch (per-channel) or
    batch*n_groups (per-group) -- returns the per-transition data reversed so the
    sweep from large positive s toward zero is a contiguous slice, plus j_min (the
    number of leading, all-negative transitions to skip).

    Returns (i_rev, delta_rev, sts_rev, k_curr_val_rev, j_min):
      i_rev          (N, T): changed weight index per transition (into 0..D-1).
      delta_rev      (N, T): signed grid delta applied at that transition.
      sts_rev        (N, T): the transition scale (descending along T).
      k_curr_val_rev (N, T): the entry's grid value *before* the transition
                             (k_curr); used by the diagonal variants.
      j_min          (int) : leading transitions that are all-negative for every
                             row and can be skipped (they cannot yield s > 0).
    """
    _, k_curr, _, delta_next_curr = _piso_grid_deltas(unscaled_grid)

    N, D = w_flat.shape

    transition_scales = 2 * w_flat.unsqueeze(1) / (unscaled_grid[:-1] +
                                                   unscaled_grid[1:]).unsqueeze(1).unsqueeze(0)
    sorted_transition_scales, scales_sorting_indices = torch.sort(
        transition_scales.reshape(N, -1), dim=1)
    del transition_scales

    grid_pair = scales_sorting_indices // D  # (N, T): which grid transition k
    all_i = scales_sorting_indices % D  # (N, T): which weight index
    all_delta_base = delta_next_curr[grid_pair]  # (N, T)
    all_k_curr = k_curr[grid_pair]  # (N, T): grid value before the transition
    del scales_sorting_indices, grid_pair

    # Skip all negative-scale transitions (they cannot yield s > 0). In the sorted
    # order the negatives come first, so j_min is one before the first non-negative.
    neg_counts = (sorted_transition_scales < 0).sum(dim=1)  # (N,)
    j_min = (neg_counts.min() - 1).item()

    # Signed deltas: positive transitions keep the sign, negative ones flip it.
    all_delta = torch.where(sorted_transition_scales >= 0, all_delta_base, -all_delta_base)
    del all_delta_base

    # Reverse so each chunk is a contiguous slice (sweep runs descending in j).
    return (
        all_i.flip(1),
        all_delta.flip(1),
        sorted_transition_scales.flip(1),
        all_k_curr.flip(1),
        j_min)


def _chunk_lower_bounds(
        sts_rev: torch.Tensor, chunk_start_idx: int, chunk_end_idx: int,
        c_len: int) -> torch.Tensor:
    """Lower interval bounds for a chunk: the next (descending) transition scale,
    with -inf for the final position of the sweep."""
    N = sts_rev.shape[0]
    lb_end = min(chunk_end_idx + 1, sts_rev.shape[1])
    lower_chunk = sts_rev[:, chunk_start_idx + 1:lb_end]
    if lower_chunk.shape[1] < c_len:
        lower_chunk = torch.cat([
            lower_chunk,
            torch.full((N, 1), -float('inf'), device=sts_rev.device, dtype=sts_rev.dtype)],
                                dim=1)
    return lower_chunk


def _fused_chunk_eval_best(
    wHq_inc: torch.Tensor,
    qHq_inc: torch.Tensor,
    upper_chunk: torch.Tensor,
    lower_chunk: torch.Tensor,
    wHq: torch.Tensor,
    qHq: torch.Tensor,
    optimal_scale: torch.Tensor,
    optimal_error: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form evaluation of all C positions in a chunk (positive scales only).

    Given the per-position increments of wHq and qHq, builds the running
    accumulators via prefix sums (cumsum) over the chunk, computes the clamped
    closed-form scale and error at each position, finds the best valid position in
    the chunk, and folds it into the running optimum. Shared by every variant; the
    caller supplies the variant-specific increments.

    All tensors are shaped (N, C) except the accumulators/optimals which are (N,).
    Returns (wHq_end, qHq_end, optimal_scale, optimal_error, improved, best_in_chunk),
    where improved (N,) marks rows whose optimum was updated by this chunk and
    best_in_chunk (N,) is the within-chunk position that won (needed by the
    sequential variants to reconstruct the winning q vector).
    """
    # Prefix sums within the chunk, added to the incoming accumulators.
    wHq_all = wHq.unsqueeze(-1) + torch.cumsum(wHq_inc, dim=-1)  # (N, C)
    qHq_all = qHq.unsqueeze(-1) + torch.cumsum(qHq_inc, dim=-1)  # (N, C)

    # Closed-form optimum per position, clamped into the interval.
    s_all = wHq_all / qHq_all
    s_all = torch.minimum(s_all, upper_chunk - eps)
    s_all = torch.maximum(s_all, lower_chunk + eps)
    error_all = s_all * s_all * qHq_all - 2.0 * s_all * wHq_all

    # Validity: finite s, improves the error, non-degenerate interval, s > 0.
    interval_width = torch.abs(lower_chunk - upper_chunk)
    update_mask = ((~s_all.isnan()) & (error_all < optimal_error.unsqueeze(-1)) &
                   (interval_width > eps) & (s_all > 0.0))

    # Best (lowest-error) valid position within the chunk.
    error_masked = torch.where(update_mask, error_all, torch.full_like(error_all, torch.inf))
    best_in_chunk = error_masked.argmin(dim=-1)
    best_error = torch.gather(error_masked, -1, best_in_chunk.unsqueeze(-1)).squeeze(-1)
    best_scale = torch.gather(s_all, -1, best_in_chunk.unsqueeze(-1)).squeeze(-1)

    improved = best_error < optimal_error
    optimal_scale = torch.where(improved, best_scale, optimal_scale)
    optimal_error = torch.where(improved, best_error, optimal_error)

    return wHq_all[..., -1], qHq_all[..., -1], optimal_scale, optimal_error, improved, best_in_chunk


def _fused_chunk_eval(
        wHq_inc: torch.Tensor,
        qHq_inc: torch.Tensor,
        upper_chunk: torch.Tensor,
        lower_chunk: torch.Tensor,
        wHq: torch.Tensor,
        qHq: torch.Tensor,
        optimal_scale: torch.Tensor,
        optimal_error: torch.Tensor,
        eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin wrapper over _fused_chunk_eval_best that drops the best-position info.

    Used by the variants that only need the optimal scale (no q reconstruction).
    Returns (wHq_end, qHq_end, optimal_scale, optimal_error).
    """
    wHq_end, qHq_end, optimal_scale, optimal_error, _, _ = _fused_chunk_eval_best(
        wHq_inc, qHq_inc, upper_chunk, lower_chunk, wHq, qHq, optimal_scale, optimal_error, eps)
    return wHq_end, qHq_end, optimal_scale, optimal_error


def _chunked_piso_sweep_per_channel_diag(
        w_b: torch.Tensor,
        H_diag: torch.Tensor,
        unscaled_grid: torch.Tensor,
        G_diag: Optional[torch.Tensor] = None,
        eps: float = 1.e-6,
        allow_negative_s: bool = False) -> torch.Tensor:
    """Chunked PiSO sweep for per-channel quantization with a diagonal Hessian.

    Exact reformulation of _piso_sweep_per_channel_diag (positive scales only).
    With H diagonal there is no Hq cross-term: (Hq)_i at a transition reduces to
    H_ii times the entry's grid value *before* the update (k_curr), a grid-derived
    constant. So qHq_inc[c] = H_ii * (2 * k_curr_c * delta_c + delta_c^2) is a pure
    per-position quantity and needs no V-cumulative or sub-Hessian.

    Args:
        w_b: weights, shape [batch, D].
        H_diag: diagonal of H, shape [D].
        G_diag: diagonal of G for the cross-activation objective, shape [D].
            If None, H_diag is used for both terms.
        unscaled_grid: sorted quantization grid with 0 included.
        eps: interval degeneracy threshold and clamping margin.
        allow_negative_s: must be False; negative scales are not supported.

    Returns:
        Optimal scale per channel, shape [batch, 1].
    """
    _reject_negative_scales(allow_negative_s)

    batch_size, D = w_b.shape
    i_rev, delta_rev, sts_rev, k_curr_rev, j_min = _chunked_transition_setup(w_b, unscaled_grid)
    T = i_rev.shape[1]

    # Hw = (G or H) elementwise * w -- fixed throughout the sweep.
    Hw = (G_diag if G_diag is not None else H_diag) * w_b  # (B, D)

    wHq = torch.zeros((batch_size,), dtype=w_b.dtype, device=w_b.device)
    qHq = torch.zeros((batch_size,), dtype=w_b.dtype, device=w_b.device)
    optimal_scale = torch.empty((batch_size,), device=w_b.device)
    optimal_error = torch.full((batch_size,), torch.inf, device=w_b.device)

    n_iterations = T - j_min
    C = _CHUNKED_PISO_CHUNK_SIZE

    for chunk_start_idx in range(0, n_iterations, C):
        chunk_end_idx = min(chunk_start_idx + C, n_iterations)
        c_len = chunk_end_idx - chunk_start_idx

        i_chunk = i_rev[:, chunk_start_idx:chunk_end_idx]  # (B, c_len)
        delta_chunk = delta_rev[:, chunk_start_idx:chunk_end_idx]  # (B, c_len)
        k_curr_chunk = k_curr_rev[:, chunk_start_idx:chunk_end_idx]  # (B, c_len)
        upper_chunk = sts_rev[:, chunk_start_idx:chunk_end_idx]  # (B, c_len)
        lower_chunk = _chunk_lower_bounds(sts_rev, chunk_start_idx, chunk_end_idx, c_len)

        H_ii_chunk = H_diag[i_chunk]  # (B, c_len)
        wHq_inc = delta_chunk * torch.gather(Hw, 1, i_chunk)
        # (Hq)_i before the update is H_ii * k_curr (the entry's current grid value).
        qHq_inc = H_ii_chunk * (2.0 * k_curr_chunk * delta_chunk + delta_chunk * delta_chunk)

        wHq, qHq, optimal_scale, optimal_error = _fused_chunk_eval(
            wHq_inc, qHq_inc, upper_chunk, lower_chunk, wHq, qHq, optimal_scale, optimal_error,
            eps)

    return optimal_scale.unsqueeze(1)


def _chunked_piso_sweep_per_group_diag(
        w_b: torch.Tensor,
        H_diag: torch.Tensor,
        unscaled_grid: torch.Tensor,
        G_diag: Optional[torch.Tensor] = None,
        eps: float = 1.e-6,
        allow_negative_s: bool = False) -> torch.Tensor:
    """Chunked PiSO sweep for group-wise quantization with a diagonal Hessian.

    Exact reformulation of _piso_sweep_per_group_diag (positive scales only).
    Combines the independent-groups flattening with the diagonal fast path: no Hq
    cross-term, qHq_inc uses the entry's current grid value (k_curr).

    Args:
        w_b: weights, shape [batch, n_groups, group_size].
        H_diag: diagonal of H, shape [D] (D = n_groups * group_size).
        G_diag: diagonal of G, shape [D]. If None, H_diag is used for both.
        unscaled_grid: sorted quantization grid with 0 included.
        eps: interval degeneracy threshold and clamping margin.
        allow_negative_s: must be False; negative scales are not supported.

    Returns:
        Optimal scale per group, shape [batch, n_groups, 1].
    """
    _reject_negative_scales(allow_negative_s)

    batch_size, n_groups, group_size = w_b.shape
    N = batch_size * n_groups
    w_flat = w_b.reshape(N, group_size)

    # Per-group slices of the diagonals, expanded over the batch: (N, gs).
    H_diag_g = H_diag.reshape(n_groups, group_size).unsqueeze(0).expand(batch_size, -1,
                                                                        -1).reshape(N, group_size)
    if G_diag is not None:
        G_diag_g = G_diag.reshape(n_groups,
                                  group_size).unsqueeze(0).expand(batch_size, -1,
                                                                  -1).reshape(N, group_size)
        Hw = G_diag_g * w_flat
    else:
        Hw = H_diag_g * w_flat

    i_rev, delta_rev, sts_rev, k_curr_rev, j_min = _chunked_transition_setup(w_flat, unscaled_grid)
    T = i_rev.shape[1]

    wHq = torch.zeros((N,), dtype=w_b.dtype, device=w_b.device)
    qHq = torch.zeros((N,), dtype=w_b.dtype, device=w_b.device)
    optimal_scale = torch.empty((N,), device=w_b.device)
    optimal_error = torch.full((N,), torch.inf, device=w_b.device)

    n_iterations = T - j_min
    C = _CHUNKED_PISO_CHUNK_SIZE

    for chunk_start_idx in range(0, n_iterations, C):
        chunk_end_idx = min(chunk_start_idx + C, n_iterations)
        c_len = chunk_end_idx - chunk_start_idx

        i_chunk = i_rev[:, chunk_start_idx:chunk_end_idx]  # (N, c_len)
        delta_chunk = delta_rev[:, chunk_start_idx:chunk_end_idx]  # (N, c_len)
        k_curr_chunk = k_curr_rev[:, chunk_start_idx:chunk_end_idx]  # (N, c_len)
        upper_chunk = sts_rev[:, chunk_start_idx:chunk_end_idx]  # (N, c_len)
        lower_chunk = _chunk_lower_bounds(sts_rev, chunk_start_idx, chunk_end_idx, c_len)

        H_ii_chunk = torch.gather(H_diag_g, 1, i_chunk)  # (N, c_len)
        wHq_inc = delta_chunk * torch.gather(Hw, 1, i_chunk)
        qHq_inc = H_ii_chunk * (2.0 * k_curr_chunk * delta_chunk + delta_chunk * delta_chunk)

        wHq, qHq, optimal_scale, optimal_error = _fused_chunk_eval(
            wHq_inc, qHq_inc, upper_chunk, lower_chunk, wHq, qHq, optimal_scale, optimal_error,
            eps)

    return optimal_scale.reshape(batch_size, n_groups, 1)


@SCALE_SOLVER_REGISTRY.register(names='chunked-piso')
class ChunkedPiSOSolverFamily(ScaleSolverFamily):
    """Chunked reformulation of the interval sweep, diagonal H; positive scales
    only (allow_negative_s=True raises ValueError).

    The dense-H slots (per_channel, per_group, groups_sequential,
    single_group_sequential) are not implemented, so the base stubs raise
    NotImplementedError.
    """

    _per_channel_diag = staticmethod(_chunked_piso_sweep_per_channel_diag)
    _per_group_diag = staticmethod(_chunked_piso_sweep_per_group_diag)
