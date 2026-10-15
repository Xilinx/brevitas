# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared building blocks for the scale-optimization solvers.

Holds the two pieces every solver family reuses:

  - the interval-sweep helpers (`_piso_grid_deltas`, `_piso_select_optimal`,
    `_piso_select_optimal_straddle`): the grid-derived transition deltas and the
    closed-form/optimality tail, factored out so they are fixed once rather than
    duplicated across the per-variant sweeps;
  - the solver registry (`SCALE_SOLVER_REGISTRY`) and the `ScaleSolverFamily`
    base class, where a family binds its sweeps and gets the layer/group
    dispatch the updaters need.

This module depends only on torch and the registry primitive; the concrete
solver modules (`piso`, `chunked_piso`) import from here, never the other way
around.

Reference: "Optimal Post-Training Quantization Scales and Where to Find Them"
(https://arxiv.org/abs/2606.10890).
"""

from abc import ABC
from typing import Callable
from typing import Optional
from typing import Tuple

import torch

from brevitas.utils.python_utils import Registry

# -----------------------------------------------------------------------------
# Interval-sweep helpers.
#
# The two identical pieces shared by every _*_sweep_* solver: the grid-derived
# transition deltas (_piso_grid_deltas) and the closed-form/optimality tail
# (_piso_select_optimal[_straddle]). The per-interval update of wHq/qHq/Hq
# differs per variant and stays inline in each solver.
# -----------------------------------------------------------------------------


def _piso_grid_deltas(
        unscaled_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grid-derived constants shared by every sweep, independent of the weights.

    Returns (idx_0, k_curr, k_next, delta_next_curr) where idx_0 is the index of
    0 in the grid and delta_next_curr[k] is the change in a q entry when crossing
    transition k (see paper: q walks the grid as s decreases).

    Batch-aware: ``unscaled_grid`` may be a single 1-D grid ``[L]`` shared by
    every row, or a local (per-channel / per-group) grid ``[..., L]`` with a
    distinct grid per leading position, as the integer-zero-point translation
    produces. The last three outputs mirror its shape with the level axis
    reduced to ``L-1``; ``idx_0`` mirrors the leading shape (a 0-d tensor for a
    1-D grid). Every row must be sorted ascending and contain exactly one 0.
    """
    if not torch.all(unscaled_grid[..., :-1] < unscaled_grid[..., 1:]):
        raise ValueError("unscaled_grid is not sorted ascending or contains duplicates")
    if not bool(torch.all(torch.any(unscaled_grid == 0, dim=-1))):
        raise ValueError("0 not in unscaled_grid")
    L = unscaled_grid.shape[-1]
    # Index of 0 in each row (grids are strictly ascending, so 0 appears once).
    idx_0 = (unscaled_grid == 0).to(torch.long).argmax(dim=-1)
    ar = torch.arange(L - 1, device=unscaled_grid.device)
    # When a q entry changes, its old and new grid values are known a priori:
    # the entry walks grid[k] -> grid[k+1] above 0, and its mirror below 0.
    # Build the two index rows and gather, batching over any leading dims.
    lower = ar < idx_0.unsqueeze(-1)
    curr_idx = torch.where(lower, ar + 1, ar).expand(*unscaled_grid.shape[:-1], L - 1)
    next_idx = torch.where(lower, ar, ar + 1).expand(*unscaled_grid.shape[:-1], L - 1)
    k_curr = torch.gather(unscaled_grid, -1, curr_idx)
    k_next = torch.gather(unscaled_grid, -1, next_idx)
    delta_next_curr = k_next - k_curr
    return idx_0, k_curr, k_next, delta_next_curr


def _piso_transition_lookup(t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Per-transition value for each row: ``t[..., k[...]]``.

    ``t`` is a grid-derived per-transition tensor (delta / k_curr / k_next),
    either shared across rows (``[L-1]``) or local to them (``[..., L-1]``).
    ``k`` holds transition indices and may carry either one index per row (the
    sequential sweeps: ``[B]`` per-channel, ``[B, n_groups]`` per-group) or many
    (the chunked family, which resolves every transition at once: ``[N, T]``
    against a ``[N, L-1]`` tensor). Returns the looked-up values with ``k``'s
    shape.

    The shared-grid case keeps the plain advanced-index it had before local grids
    existed; for the sequential sweeps this runs once per swept interval, and
    routing it through the gather costs about twice as much per call.
    """
    if t.dim() == 1:
        return t[k]
    if k.dim() == t.dim():
        # one lookup per (row, transition): indices already match t's rank
        return torch.gather(t, -1, k)
    return torch.gather(t, -1, k.unsqueeze(-1)).squeeze(-1)


def _piso_sorted_transitions(
        w_b: torch.Tensor,
        unscaled_grid: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorted transition scales, their flat indices, and the shifted weights.

    The scales at which one entry of q changes are ``2 (w_i + z) / (g_k +
    g_{k+1})``: a zero point shifts the breakpoints (q flips on ``(w + z)/s``)
    but not the linear term, so the shifted weights come back for the caller's
    sign-change state. With no zero point they are ``w_b`` itself, at no cost.

    Handles per-channel weights ``[B, D]`` and per-group weights ``[B, n_groups,
    G]``, against a grid shared by every row (``[L]``) or local to it
    (``[..., L]``). The transitions are flattened in ``(L-1, last axis)`` order,
    so a sorted index still decodes as ``k = idx // D``, ``i = idx % D``.

    Returns (sorted_transition_scales, sorting_indices, w_shift); the sweep's
    interval count is ``sorted_transition_scales.shape[-1]``.
    """
    mids = (unscaled_grid[..., :-1] + unscaled_grid[..., 1:]).unsqueeze(-1)
    w_shift = w_b if zero_point is None else w_b + zero_point
    transitions = 2 * w_shift.unsqueeze(-2) / mids
    sorted_scales, sorting_indices = torch.sort(transitions.flatten(-2), dim=-1)
    return sorted_scales, sorting_indices, w_shift


def translate_grid_by_zero_point(
        unscaled_grid: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Translate an unsigned grid so a symmetric sweep solves the *asymmetric*
    objective of a quantizer with a constant integer zero point.

    Brevitas asymmetric integer quantization dequantizes a weight as

        w_hat = s * (clamp(round(w / s + z), 0, M) - z),   M = 2^bit_width - 1,

    for scale ``s`` and integer zero point ``z``. Since ``round(y + z) =
    round(y) + z`` for integer ``z`` (off the half-integer ties, which are
    exactly the transition scales the sweep already treats by its endpoint
    convention),

        w_hat = s * clamp(round(w / s), -z, M - z) = s * q,  q in {0..M} - z,

    i.e. the *symmetric* quantizer on the translated grid. The weights are left
    untouched; only the grid moves, so no solver term changes.

    A per-channel or per-group ``z`` produces a *local* grid, one row per scale
    entry, which the non-sequential sweeps consume directly. Pass ``z`` in the
    scale's shape (``[B, 1]`` per-channel, ``[B, n_groups, 1]`` per-group) so
    that ``{0..M} - z`` broadcasts to the ``[B, L]`` / ``[B, n_groups, L]`` grid.

    Args:
        unscaled_grid: sorted-ascending 1-D grid, the unsigned native grid with
            ``grid[0] == 0``.
        zero_point: integer zero point ``z``, a scalar (uniform translation) or
            a tensor broadcastable against ``unscaled_grid``.

    Returns:
        The translated grid, which still contains 0 and is still ascending.
    """
    if not torch.all(unscaled_grid[:-1] < unscaled_grid[1:]):
        raise ValueError("unscaled_grid must be sorted ascending without duplicates.")
    if unscaled_grid[0] != 0:
        raise ValueError(
            "translate_grid_by_zero_point expects the unsigned native grid with "
            f"grid[0] == 0; got grid[0] = {unscaled_grid[0].item()}.")
    z = torch.as_tensor(zero_point, dtype=unscaled_grid.dtype, device=unscaled_grid.device)
    if not bool(torch.all(torch.round(z) == z)):
        raise ValueError("zero_point must be integer-valued.")
    if bool(torch.any((z < 0) | (z > unscaled_grid[-1]))):
        raise ValueError(
            f"zero_point out of [0, {unscaled_grid[-1].item()}]; the translated "
            "grid would no longer contain 0.")
    return unscaled_grid - z


def _piso_select_optimal(
        num: torch.Tensor,
        qHq: torch.Tensor,
        lower_limit: torch.Tensor,
        upper_limit: torch.Tensor,
        mask: torch.Tensor,
        optimal_scale: torch.Tensor,
        optimal_error: torch.Tensor,
        eps: float,
        allow_negative_s: bool) -> Tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Closed-form minimizer + running-optimal update for one swept interval.

    Shared tail of every sweep. Given the interval's numerator num (= alpha),
    qHq (= beta) and bounds, computes the clamped minimizer s* = num/qHq, its
    error, and folds it into optimal_scale/optimal_error where it improves.

    Returns (optimal_scale, optimal_error, should_break, update_mask).
    should_break is True when only positive scales are allowed and the sweep
    has entered the all-negative region, so the caller can stop iterating.
    update_mask (which rows were improved this interval) is returned for the
    sequential-groups sweep, which also uses it to commit the group's q vector;
    the other sweeps ignore it.
    """
    # Closed form s* = alpha / beta, clamped into the interval bounds.
    s_current = num / qHq
    s_current = torch.minimum(s_current, upper_limit - eps)
    s_current = torch.maximum(s_current, lower_limit + eps)

    error_current = (s_current ** 2) * qHq - 2 * s_current * num

    update_mask = (
        # the scale should not be nan
        (~s_current.isnan())
        # there should be an error improvement
        & (error_current < optimal_error)
        # skip degenerate (zero-length) intervals: several q entries change at
        # once (weights multiple of each other) and the state is not yet coherent
        & (torch.abs(lower_limit - upper_limit) > eps))

    should_break = False
    if not allow_negative_s:
        # A positive-only optimum can never come from a negative candidate, so
        # reject them regardless of whether we also stop iterating this step.
        update_mask &= (s_current > 0)
        # Sweep runs from large to small s; once every interval is negative there
        # is nothing left to explore for a positive-only scale.
        if torch.all(mask):
            should_break = True

    optimal_scale = torch.where(update_mask, s_current, optimal_scale)
    optimal_error = torch.where(update_mask, error_current, optimal_error)
    return optimal_scale, optimal_error, should_break, update_mask


def _piso_select_optimal_straddle(
        num_transition: torch.Tensor,
        qHq_transition: torch.Tensor,
        lower_limit: torch.Tensor,
        upper_limit: torch.Tensor,
        mask: torch.Tensor,
        optimal_scale: torch.Tensor,
        optimal_error: torch.Tensor,
        eps: float,
        allow_negative_s: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the negative half [lower, 0] of the zero-straddling interval.

    The interval that straddles zero (lower < 0 < upper) is the only one on
    which q is not constant: every entry flips as s crosses 0 (q = clamp(
    round(w/s))). The normal _piso_select_optimal call covers its positive half
    [0, upper] with the running state; its negative half [lower, 0] uses the
    post-sign-change state (q_transition), so evaluate that explicitly here or
    the negative optimum inside (lower, 0) would be missed. Only relevant when
    negative scales are searched.

    Returns the updated (optimal_scale, optimal_error).
    """
    if not allow_negative_s:
        return optimal_scale, optimal_error
    straddles_zero = (lower_limit < 0) & (upper_limit > 0)
    if torch.any(straddles_zero):
        # Restrict to the straddling entries: elsewhere collapse the interval to
        # [0, 0] so the degeneracy guard rejects any update.
        neg_lower = torch.where(straddles_zero, lower_limit, torch.zeros_like(lower_limit))
        neg_upper = torch.zeros_like(upper_limit)
        optimal_scale, optimal_error, _, _ = _piso_select_optimal(
            num_transition, qHq_transition, neg_lower, neg_upper, mask, optimal_scale,
            optimal_error, eps, allow_negative_s)
    return optimal_scale, optimal_error


# -----------------------------------------------------------------------------
# Solver families.
#
# A ScaleSolverFamily subclass groups the sweeps for one scale-optimization
# algorithm and exposes the two dispatches the updaters need: one per layer, one
# per group. ScaleOptimizer resolves a family by name from SCALE_SOLVER_REGISTRY,
# so selecting or adding a solver is a registry lookup, not an updater change.
#
# To add one: implement the sweeps in a new module, bind them to the slots of a
# ScaleSolverFamily subclass, and register that class via
# SCALE_SOLVER_REGISTRY.register. Families are registered as classes, never
# instantiated.
# -----------------------------------------------------------------------------

SCALE_SOLVER_REGISTRY = Registry("SCALE_SOLVER_REGISTRY")


class ScaleSolverFamily(ABC):
    """A named set of scale-optimization sweeps + their layer/group dispatch.

    A family overrides the six static solver slots, usually by binding an
    existing sweep (``_per_channel_diag = staticmethod(_piso_sweep_per_channel_diag)``).
    Base slots raise NotImplementedError, so a family may implement only a
    subset: an unimplemented slot still dispatches and fails when called, not
    when selected. Slots take *args/**kwargs so a family can take extras.

    get_layer_solver returns (solver_fn, arg_kind) for a whole layer,
    get_group_solver returns (solver_fn, tag) for one group. arg_kind
    ('dense'|'diag'|'greedy') and tag ('group-dense'|'group-diag'|
    'group-greedy') tell the caller which H/G kwargs to build. Pure dispatch:
    no tensors are touched.

    Every family implements both asymmetric extensions: the sweeps accept a
    local (per-row) grid, which carries an integer zero point, and a
    ``zero_point`` argument, which carries a free offset.
    """

    # -- solver slots ---------------------------------------------------------

    @staticmethod
    def _per_channel(
            w_b: torch.Tensor,
            H: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("per-channel sweep not implemented by this family.")

    @staticmethod
    def _per_channel_diag(
            w_b: torch.Tensor,
            H_diag: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G_diag: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("diagonal per-channel sweep not implemented by this family.")

    @staticmethod
    def _per_group(
            w_b: torch.Tensor,
            H: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("per-group sweep not implemented by this family.")

    @staticmethod
    def _per_group_diag(
            w_b: torch.Tensor,
            H_diag: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G_diag: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("diagonal per-group sweep not implemented by this family.")

    @staticmethod
    def _groups_sequential(
            w_b: torch.Tensor,
            H: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("sequential groups sweep not implemented by this family.")

    @staticmethod
    def _single_group_sequential(
            w_orig: torch.Tensor,
            w_updated: torch.Tensor,
            H: torch.Tensor,
            unscaled_grid: torch.Tensor,
            group_size: int,
            group_idx: int,
            G: Optional[torch.Tensor] = None,
            eps: float = 1.e-6,
            allow_negative_s: bool = False,
            *args,
            **kwargs) -> torch.Tensor:
        raise NotImplementedError("single sequential group sweep not implemented by this family.")

    # -- dispatch -------------------------------------------------------------

    @classmethod
    def get_layer_solver(cls, is_group: bool, use_diag: bool,
                         is_greedy: bool) -> Tuple[Callable[..., torch.Tensor], str]:
        if is_greedy:
            return cls._groups_sequential, 'greedy'
        dispatch = {
            (True, True): (cls._per_group_diag, 'diag'),
            (True, False): (cls._per_group, 'dense'),
            (False, True): (cls._per_channel_diag, 'diag'),
            (False, False): (cls._per_channel, 'dense'),}
        return dispatch[(is_group, use_diag)]

    @classmethod
    def get_group_solver(cls, use_diag: bool,
                         per_group_greedy: bool) -> Tuple[Callable[..., torch.Tensor], str]:
        if use_diag:
            return cls._per_channel_diag, 'group-diag'
        if per_group_greedy:
            return cls._single_group_sequential, 'group-greedy'
        return cls._per_channel, 'group-dense'
