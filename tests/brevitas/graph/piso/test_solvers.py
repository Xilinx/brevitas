# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PiSO interval-sweep solvers (brevitas.graph.piso.solvers).

Each solver returns, in closed form, the weight-quantization scale that minimizes
the reconstruction objective

    E(s) = s^2 * (q^T H q) - 2 s * (q^T G w),     q = clamp(round(w / s), grid)

(with G defaulting to H). This is the scale-dependent part of
||X w - s X_tilde q||^2 up to the s-independent constant w^T X^T X w.

These tests evaluate that objective directly and compare each solver's result
against a brute-force grid search over candidate scales. Because PiSO is exact,
the solver's objective must be no larger than the best objective found by grid
search (up to a small tolerance).

Solver-agnostic per-channel tests are parametrized over the available per-channel
dense solvers (the sequential 'piso' and the 'chunked-piso' reformulation); the
chunked solver is skipped on cases that require negative scales, since it only
supports positive scales (symmetric grids).
"""
import math

import pytest
import torch

from brevitas.graph.piso.solvers import ChunkedPiSOSolverFamily
from brevitas.graph.piso.solvers import PiSOSolverFamily
from brevitas.graph.piso.solvers import ScaleSolverFamily
from brevitas.graph.piso.solvers import translate_grid_by_zero_point
from brevitas.graph.piso.solvers.common import _piso_grid_deltas
from brevitas.graph.piso.solvers.common import build_quant_scale_problem
from brevitas.graph.piso.utils import make_fp_grid
from brevitas.graph.piso.utils import make_int_grid

# The sweeps are reached through the family slots rather than imported directly,
# so a slot that is not implemented resolves to the ScaleSolverFamily stub (which
# raises NotImplementedError when called) instead of failing at import.
_piso_sweep_per_channel = PiSOSolverFamily._per_channel
_piso_sweep_per_channel_diag = PiSOSolverFamily._per_channel_diag
_piso_sweep_per_group = PiSOSolverFamily._per_group
_piso_sweep_per_group_diag = PiSOSolverFamily._per_group_diag
_piso_sweep_groups_sequential = PiSOSolverFamily._groups_sequential
_piso_sweep_single_group_sequential = PiSOSolverFamily._single_group_sequential
_chunked_piso_sweep_per_channel = ChunkedPiSOSolverFamily._per_channel
_chunked_piso_sweep_per_channel_diag = ChunkedPiSOSolverFamily._per_channel_diag
_chunked_piso_sweep_per_group = ChunkedPiSOSolverFamily._per_group
_chunked_piso_sweep_per_group_diag = ChunkedPiSOSolverFamily._per_group_diag
_chunked_piso_sweep_groups_sequential = ChunkedPiSOSolverFamily._groups_sequential
_chunked_piso_sweep_single_group_sequential = ChunkedPiSOSolverFamily._single_group_sequential


def _not_implemented(solver_fn):
    """True while solver_fn is still the ScaleSolverFamily stub."""
    return getattr(solver_fn, '__qualname__', '').startswith('ScaleSolverFamily.')


# The dense-H solvers are not implemented. The mark is conditional, so it goes
# inert on its own once they are: no test needs to change then.
DENSE_NOT_IMPLEMENTED = _not_implemented(PiSOSolverFamily._per_channel)
dense_xfail = pytest.mark.xfail(
    DENSE_NOT_IMPLEMENTED,
    raises=NotImplementedError,
    strict=True,
    reason="dense solvers are not implemented")

# For tests that exercised the dense and the diagonal variant in one body: the
# variant becomes a parameter so the diagonal one keeps running on its own.
DENSE_DIAG_PARAMS = [pytest.param('dense', marks=[dense_xfail]), 'diag']


def dense_params(mapping):
    """Parametrize over solver names, xfailing the ones backed by a dense slot.

    Values are either the solver itself or, for RECOVERY_SOLVERS, a
    (runner, "dense"|"diag") pair whose tag names the slot family.
    """
    params = []
    for name, value in mapping.items():
        is_dense = value[1] == 'dense' if isinstance(value, tuple) else _not_implemented(value)
        params.append(pytest.param(name, marks=[dense_xfail]) if is_dense else name)
    return params


DTYPE = torch.float64  # high precision so "optimal" comparisons are meaningful

# PiSO clamps its closed-form minimizer by eps inside each interval, so it can
# miss the exact vertex by ~eps; run the solvers with a tiny eps so that clamping
# error is negligible. The grid-search reference also has a small residual
# coarseness (~1e-5 absolute; it converges as n_candidates grows), so compare
# objectives with a tolerance that comfortably covers both.
SOLVER_EPS = 1e-9
N_GRID_CANDIDATES = 8000
ATOL = 1e-5
RTOL = 1e-6


def objective_leq(E_solver, E_grid):
    """PiSO objective must be <= grid-search objective (up to a small slack)."""
    return bool(torch.all(E_solver <= E_grid + ATOL + RTOL * E_grid.abs()))


# -----------------------------------------------------------------------------
# Objective evaluation (shared reference, independent of the solver internals).
# -----------------------------------------------------------------------------
def quantize_to_grid(w: torch.Tensor, s: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """q = round-to-nearest grid value of (w / s), clamped to [grid[0], grid[-1]].

    w: [..., D], s: [...] (broadcast over the last dim), grid: [G] sorted ascending.
    Matches the assignment the solvers sweep: an entry switches from grid[k] to
    grid[k+1] exactly when w/s crosses the midpoint (grid[k]+grid[k+1])/2.
    """
    x = w / s.unsqueeze(-1)
    # nearest grid index per element via searchsorted on the midpoints.
    # Flatten to 1D for an unambiguous searchsorted, then restore the shape.
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    flat = x.reshape(-1).contiguous()
    idx = torch.searchsorted(midpoints, flat)
    idx = idx.clamp(0, grid.numel() - 1)
    return grid[idx].reshape(x.shape)


def eval_dense(w, s, H, grid, G=None):
    """Objective for the dense-H per-channel case. w:[B,D], s:[B]."""
    G = H if G is None else G
    q = quantize_to_grid(w, s, grid)
    Hq = (H @ q.T).T
    qHq = (q * Hq).sum(-1)
    Gw = (G @ w.T).T
    qGw = (q * Gw).sum(-1)
    return s ** 2 * qHq - 2 * s * qGw


def eval_diag(w, s, h_diag, grid, g_diag=None):
    """Objective for the diagonal-H per-channel case. w:[B,D], s:[B], h_diag:[D]."""
    g_diag = h_diag if g_diag is None else g_diag
    q = quantize_to_grid(w, s, grid)
    qHq = (q * h_diag * q).sum(-1)
    qGw = (q * g_diag * w).sum(-1)
    return s ** 2 * qHq - 2 * s * qGw


def grid_search_scales(w, eval_fn, grid, allow_negative_s, n_candidates=N_GRID_CANDIDATES):
    """Brute-force best scale per row over a dense sweep of candidate scales.

    Candidate scales are drawn from the union of:
      - a dense linear sweep spanning the plausible range, and
      - every per-entry transition scale 2 w_i / (grid[k] + grid[k+1]) (the exact
        breakpoints where the assignment q changes), plus small offsets so we also
        probe interval interiors.
    Returns (best_s [B], best_E [B]).
    """
    B, D = w.shape
    device, dtype = w.device, w.dtype
    # Range: scales that map the largest |w| to roughly the grid extremes.
    max_abs_w = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)  # [B,1]
    grid_max = grid.abs().max()
    s_hi = (max_abs_w / grid_max * 4.0)  # generous upper bound per row
    lin = torch.linspace(0.0, 1.0, n_candidates, device=device, dtype=dtype).unsqueeze(0)  # [1,N]
    pos = lin * s_hi  # [B,N]
    candidates = [pos]
    if allow_negative_s:
        candidates.append(-pos)
    # Exact transition breakpoints and near-interior probes.
    mids = (grid[:-1] + grid[1:]) / 2.0  # [G-1]
    mids = mids[mids != 0]
    trans = 2 * w.unsqueeze(-1) / mids.view(1, 1, -1)  # [B,D,G-1]
    trans = trans.reshape(B, -1)
    candidates.append(trans)
    candidates.append(trans * (1 + 1e-4))
    candidates.append(trans * (1 - 1e-4))
    s_cand = torch.cat(candidates, dim=1)  # [B, M]
    if not allow_negative_s:
        s_cand = torch.where(s_cand > 0, s_cand, torch.full_like(s_cand, float('nan')))
    # Evaluate objective for every candidate, per row.
    M = s_cand.shape[1]
    E = torch.full((B, M), float('inf'), device=device, dtype=dtype)
    for m in range(M):
        s_m = s_cand[:, m]
        valid = ~torch.isnan(s_m) & (s_m.abs() > 0)
        if valid.any():
            e = eval_fn(w, torch.where(valid, s_m, torch.ones_like(s_m)), grid)
            E[:, m] = torch.where(valid, e, torch.full_like(e, float('inf')))
    best_idx = E.argmin(dim=1)
    best_s = s_cand[torch.arange(B, device=device), best_idx]
    best_E = E[torch.arange(B, device=device), best_idx]
    return best_s, best_E


# -----------------------------------------------------------------------------
# Grids under test.
# -----------------------------------------------------------------------------
def int_grid(bit_width, signed=True, narrow_range=False):
    g = make_int_grid(bit_width=bit_width, signed=signed, narrow_range=narrow_range)
    return torch.tensor(g, dtype=DTYPE)


def fp_grid(e_bits, m_bits, signed=True):
    g = make_fp_grid(e_bits=e_bits, m_bits=m_bits, signed=signed)
    return torch.tensor(g, dtype=DTYPE)


GRIDS = {
    "int4_signed": int_grid(4, signed=True),
    "int3_signed": int_grid(3, signed=True),
    "int4_narrow": int_grid(4, signed=True, narrow_range=True),
    "fp_e2m1": fp_grid(2, 1),
    "fp_e3m2": fp_grid(3, 2),}


def spd_matrix(D, seed):
    """Random symmetric positive-definite [D, D]."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g, dtype=DTYPE)
    return A @ A.T + D * torch.eye(D, dtype=DTYPE)


def random_weights(B, D, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, D, generator=g, dtype=DTYPE)


# For an asymmetric grid the solver may search negative scales; symmetric grids not.
def allows_negative(grid):
    return bool((grid[-1] != -grid[0]).item())


# -----------------------------------------------------------------------------
# Per-channel dense solvers under a common interface.
#
# Both the sequential 'piso' and the 'chunked-piso' per-channel solvers share the
# signature (w_b, H, unscaled_grid, G, eps, allow_negative_s) -> [B, 1], so
# solver-agnostic per-channel tests parametrize over both. The chunked solver
# only supports positive scales, so tests must skip it whenever the case requires
# allow_negative_s=True (asymmetric grids); use skip_if_unsupported for that.
# -----------------------------------------------------------------------------
PER_CHANNEL_DENSE_SOLVERS = {
    "piso": _piso_sweep_per_channel,
    "chunked-piso": _chunked_piso_sweep_per_channel,}

PER_CHANNEL_DIAG_SOLVERS = {
    "piso": _piso_sweep_per_channel_diag,
    "chunked-piso": _chunked_piso_sweep_per_channel_diag,}

PER_GROUP_DENSE_SOLVERS = {
    "piso": _piso_sweep_per_group,
    "chunked-piso": _chunked_piso_sweep_per_group,}

PER_GROUP_DIAG_SOLVERS = {
    "piso": _piso_sweep_per_group_diag,
    "chunked-piso": _chunked_piso_sweep_per_group_diag,}


def skip_if_unsupported(solver_name, allow_negative_s):
    """Skip the chunked solver on cases that need negative scales."""
    if solver_name == "chunked-piso" and allow_negative_s:
        pytest.skip("chunked-piso only supports positive scales (allow_negative_s=False)")


# -----------------------------------------------------------------------------
# Tests: per-channel, dense H.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("solver_name", dense_params(PER_CHANNEL_DENSE_SOLVERS))
@pytest.mark.parametrize("grid_name", list(GRIDS))
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
@pytest.mark.parametrize("search_neg_scales", [False, True])
def test_per_channel_dense_matches_grid_search(solver_name, grid_name, D, use_G, search_neg_scales):
    # Search negative scales only when both requested and the grid is asymmetric;
    # the chunked solver cannot, so skip it whenever negative scales are searched.
    skip_if_unsupported(solver_name, search_neg_scales)
    grid = GRIDS[grid_name]
    neg = search_neg_scales and allows_negative(grid)
    solver = PER_CHANNEL_DENSE_SOLVERS[solver_name]
    B = 16
    w = random_weights(B, D, seed=1)
    H = spd_matrix(D, seed=2)
    G = spd_matrix(D, seed=3) if use_G else None

    s_solver = solver(
        w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=neg).squeeze(1)

    def eval_fn(ww, ss, gg):
        return eval_dense(ww, ss, H, gg, G=G)

    E_solver = eval_dense(w, s_solver, H, grid, G=G)
    _, E_grid = grid_search_scales(w, eval_fn, grid, allow_negative_s=neg)

    # PiSO is exact: its objective must be <= grid search's best (up to tolerance).
    assert objective_leq(E_solver, E_grid), (
        f"{solver_name} {grid_name} D={D} use_G={use_G} search_neg={search_neg_scales}: "
        f"max excess {(E_solver - E_grid).max().item():.3e}")


# -----------------------------------------------------------------------------
# Tests: per-channel, diagonal H.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("solver_name", list(PER_CHANNEL_DIAG_SOLVERS))
@pytest.mark.parametrize("grid_name", list(GRIDS))
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
def test_per_channel_diag_matches_grid_search(solver_name, grid_name, D, use_G):
    grid = GRIDS[grid_name]
    neg = allows_negative(grid)
    skip_if_unsupported(solver_name, neg)
    solver = PER_CHANNEL_DIAG_SOLVERS[solver_name]
    B = 16
    w = random_weights(B, D, seed=4)
    h_diag = spd_matrix(D, seed=5).diag().contiguous()
    g_diag = spd_matrix(D, seed=6).diag().contiguous() if use_G else None

    s_solver = solver(
        w_b=w,
        H_diag=h_diag,
        unscaled_grid=grid,
        G_diag=g_diag,
        eps=SOLVER_EPS,
        allow_negative_s=neg).squeeze(1)

    def eval_fn(ww, ss, gg):
        return eval_diag(ww, ss, h_diag, gg, g_diag=g_diag)

    E_solver = eval_diag(w, s_solver, h_diag, grid, g_diag=g_diag)
    _, E_grid = grid_search_scales(w, eval_fn, grid, allow_negative_s=neg)

    assert objective_leq(E_solver, E_grid), (
        f"{solver_name} {grid_name} D={D} use_G={use_G}: "
        f"max excess {(E_solver - E_grid).max().item():.3e}")


# -----------------------------------------------------------------------------
# Cross-check: dense-H solver == diagonal-H solver when H is diagonal.
# -----------------------------------------------------------------------------
@dense_xfail
@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_dense_equals_diag_when_H_is_diagonal(grid_name):
    grid = GRIDS[grid_name]
    B, D = 16, 6
    w = random_weights(B, D, seed=7)
    h_diag = spd_matrix(D, seed=8).diag().contiguous()
    H = torch.diag(h_diag)
    neg = allows_negative(grid)

    s_dense = _piso_sweep_per_channel(
        w_b=w, H=H, unscaled_grid=grid, allow_negative_s=neg).squeeze(1)
    s_diag = _piso_sweep_per_channel_diag(
        w_b=w, H_diag=h_diag, unscaled_grid=grid, allow_negative_s=neg).squeeze(1)

    # Compare objectives (scales may tie inside an interval); objectives must match.
    E_dense = eval_dense(w, s_dense, H, grid)
    E_diag = eval_diag(w, s_diag, h_diag, grid)
    assert torch.allclose(E_dense, E_diag, atol=1e-8, rtol=0)


# -----------------------------------------------------------------------------
# Tests: per-group, block-diagonal H (independent groups).
# The per-group objective is the per-channel objective applied independently to
# each group's slice using only that group's diagonal H block.
# -----------------------------------------------------------------------------
def eval_group_dense(w_bg, s_bg, H, grid, group_size, G=None):
    """Objective per (row, group). w_bg:[B,ng,gs], s_bg:[B,ng], H:[D,D].

    Mirrors the solver's per-group block extraction exactly:
      qHq  = q^T H_block q          (H_block: diagonal block of H)
      qGw  = q^T G_block^T w        (G_block: diagonal block of G.T; the solver
                                     computes Hw = w @ G_block, so wHq = q^T G_block^T w)
    """
    B, ng, gs = w_bg.shape
    G = H if G is None else G
    # per-group diagonal blocks (identical to the solver's extraction)
    Hb = H.reshape(ng, gs, ng, gs).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # [ng,gs,gs]
    Gb = G.T.reshape(ng, gs, ng, gs).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # G.T blocks
    q = quantize_to_grid(w_bg, s_bg, grid)  # [B,ng,gs]
    Hq = torch.einsum('gij,bgj->bgi', Hb, q)
    qHq = (q * Hq).sum(-1)  # [B,ng]
    # Hw[b,g,j] = sum_i w[b,g,i] Gb[g,i,j]  -> wHq = q^T Gb^T w  (note 'gji' below)
    Gw = torch.einsum('gji,bgj->bgi', Gb, w_bg)
    qGw = (q * Gw).sum(-1)  # [B,ng]
    return s_bg ** 2 * qHq - 2 * s_bg * qGw


@pytest.mark.parametrize("solver_name", dense_params(PER_GROUP_DENSE_SOLVERS))
@pytest.mark.parametrize("grid_name", ["int4_signed", "int3_signed", "fp_e2m1"])
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
@pytest.mark.parametrize("use_G", [False, True])
def test_per_group_matches_grid_search(solver_name, grid_name, n_groups, group_size, use_G):
    grid = GRIDS[grid_name]
    neg = allows_negative(grid)
    skip_if_unsupported(solver_name, neg)
    solver = PER_GROUP_DENSE_SOLVERS[solver_name]
    B = 12
    D = n_groups * group_size
    w = random_weights(B, D, seed=11).reshape(B, n_groups, group_size)
    H = spd_matrix(D, seed=12)
    G = spd_matrix(D, seed=13) if use_G else None

    s_solver = solver(
        w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS,
        allow_negative_s=neg).squeeze(-1)  # [B,ng]

    E_solver = eval_group_dense(w, s_solver, H, grid, group_size, G=G)  # [B,ng]

    # Grid-search each group independently by reusing the per-channel search on
    # the group slice with that group's diagonal H block. The G term uses the
    # same block convention as the solver / eval_group_dense: qGw = q^T Gb^T w.
    Hb = H.reshape(n_groups, group_size, n_groups, group_size).diagonal(
        dim1=0, dim2=2).permute(2, 0, 1)
    Gb = (G if G is not None else H).T.reshape(n_groups, group_size, n_groups, group_size).diagonal(
        dim1=0, dim2=2).permute(2, 0, 1)
    for g in range(n_groups):
        wg = w[:, g, :]  # [B, gs]
        Hg = Hb[g]  # [gs, gs]
        Gg = Gb[g]

        def eval_fn(ww, ss, gr, Hg=Hg, Gg=Gg):
            q = quantize_to_grid(ww, ss, gr)
            qHq = (q * (Hg @ q.T).T).sum(-1)
            qGw = (q * (Gg.T @ ww.T).T).sum(-1)  # q^T Gg^T w
            return ss ** 2 * qHq - 2 * ss * qGw

        _, E_grid_g = grid_search_scales(wg, eval_fn, grid, allow_negative_s=neg)
        assert objective_leq(E_solver[:, g], E_grid_g), (
            f"{solver_name} {grid_name} ng={n_groups} gs={group_size} use_G={use_G} group={g}: "
            f"max excess {(E_solver[:, g] - E_grid_g).max().item():.3e}")


@dense_xfail
@pytest.mark.parametrize("grid_name", ["int4_signed", "fp_e2m1"])
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
def test_per_group_diag_equals_per_group_when_H_diagonal(grid_name, n_groups, group_size):
    grid = GRIDS[grid_name]
    B = 12
    D = n_groups * group_size
    w = random_weights(B, D, seed=21).reshape(B, n_groups, group_size)
    h_diag = spd_matrix(D, seed=22).diag().contiguous()
    H = torch.diag(h_diag)
    neg = allows_negative(grid)

    s_group = _piso_sweep_per_group(
        w_b=w, H=H, unscaled_grid=grid, allow_negative_s=neg).squeeze(-1)
    s_group_diag = _piso_sweep_per_group_diag(
        w_b=w, H_diag=h_diag, unscaled_grid=grid, allow_negative_s=neg).squeeze(-1)

    E_group = eval_group_dense(w, s_group, H, grid, group_size)
    E_group_diag = eval_group_dense(w, s_group_diag, H, grid, group_size)
    assert torch.allclose(E_group, E_group_diag, atol=1e-8, rtol=0)


# -----------------------------------------------------------------------------
# Sanity: the closed-form scale actually quantizes into a consistent grid range.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("solver_name", dense_params(PER_CHANNEL_DENSE_SOLVERS))
@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_solver_scale_is_finite_and_positive_when_required(solver_name, grid_name):
    grid = GRIDS[grid_name]
    neg = allows_negative(grid)
    skip_if_unsupported(solver_name, neg)
    solver = PER_CHANNEL_DENSE_SOLVERS[solver_name]
    B, D = 8, 6
    w = random_weights(B, D, seed=31)
    H = spd_matrix(D, seed=32)
    s = solver(w_b=w, H=H, unscaled_grid=grid, allow_negative_s=neg).squeeze(1)
    assert torch.all(torch.isfinite(s))
    if not neg:
        assert torch.all(s > 0)


# -----------------------------------------------------------------------------
# Regression: allow_negative_s=False must reject negative scales even on
# asymmetric grids. The sweep enters a fully-negative region as s -> 0; on the
# last step (where every row's interval is negative) the closed-form minimizer
# can be negative, and it must NOT be committed. Asymmetric grids (whose natural
# use would set allow_negative_s=True) exercise this path when a caller forces
# positive-only scales -- exactly what the chunked solver relies on.
# -----------------------------------------------------------------------------
@dense_xfail
@pytest.mark.parametrize("grid_name", ["int4_signed", "int3_signed"])
@pytest.mark.parametrize("D", [4, 8, 9])
@pytest.mark.parametrize("use_G", [False, True])
def test_positive_only_rejects_negative_scale_on_asymmetric_grid(grid_name, D, use_G):
    grid = GRIDS[grid_name]
    assert allows_negative(grid)  # sanity: these grids are asymmetric
    B = 16
    # Sweep several seeds: the negative-scale-on-last-interval path only fires
    # for some weight/Hessian draws, so a single fixed seed is a weak guard.
    for seed in range(16):
        w = random_weights(B, D, seed=seed)
        H = spd_matrix(D, seed=seed + 100)
        G = spd_matrix(D, seed=seed + 200) if use_G else None

        s_dense = _piso_sweep_per_channel(
            w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False).squeeze(1)
        s_diag = _piso_sweep_per_channel_diag(
            w_b=w,
            H_diag=torch.diagonal(H),
            unscaled_grid=grid,
            G_diag=torch.diagonal(G) if G is not None else None,
            eps=SOLVER_EPS,
            allow_negative_s=False).squeeze(1)

        for name, s in (("dense", s_dense), ("diag", s_diag)):
            assert torch.all(torch.isfinite(s))
            assert torch.all(s > 0), (
                f"{name} {grid_name} D={D} use_G={use_G} seed={seed}: "
                f"non-positive scale {s[s <= 0].tolist()}")


# -----------------------------------------------------------------------------
# Minimal reproducer: W = a * Q with H = I.
#
# Take a fixed on-grid vector Q and set w = a * Q for a small *negative* a. With
# H = G = I, scaling w by s = a reconstructs w exactly (q = round(w/a) = Q), so
# the global optimum is s = a with objective E(a) = -a^2 * (Q . Q) and zero
# reconstruction error. Choosing Q with mixed signs (here [-8, 7]) makes the
# assignment flip across s = 0, and choosing a in the interval straddling zero
# puts the exact optimum on its negative half (lower, 0). A sweep that folds that
# half into the positive side never tries q = Q at s = a and misses it entirely.
#
# a = -0.1, Q = [-8, 7]  ->  w = [0.8, -0.7], straddling interval ~ (-0.107, 0.093),
# so a = -0.1 lies in (lower, 0); optimum s = -0.1, E = -0.01 * 113 = -1.13.
# -----------------------------------------------------------------------------
_MIN_GRID = GRIDS["int4_signed"]  # [-8..7], asymmetric
_MIN_Q = torch.tensor([-8.0, 7.0], dtype=DTYPE)  # on-grid, mixed sign
_MIN_A = -0.1  # target scale: negative, inside the zero-straddling interval
_MIN_W = (_MIN_A * _MIN_Q).unsqueeze(0)  # [[0.8, -0.7]]
_MIN_H = torch.eye(2, dtype=DTYPE)
_MIN_H_DIAG = torch.ones(2, dtype=DTYPE)
# Exact optimum: s = a reconstructs w, E = a^2 (Q.Q) - 2 a (a Q.Q) = -a^2 (Q.Q).
_MIN_E_OPT = -(_MIN_A ** 2) * (_MIN_Q @ _MIN_Q)


@dense_xfail
def test_per_channel_wq_identity_first_negative_interval_optimal():
    s_solver = _piso_sweep_per_channel(
        w_b=_MIN_W, H=_MIN_H, unscaled_grid=_MIN_GRID, eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(1)
    E_solver = eval_dense(_MIN_W, s_solver, _MIN_H, _MIN_GRID)

    assert s_solver.item() < 0
    # The exact minimizer is s = a; the solver must recover it (up to eps clamping).
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"per_channel W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}")


def test_per_channel_diag_wq_identity_first_negative_interval_optimal():
    s_solver = _piso_sweep_per_channel_diag(
        w_b=_MIN_W,
        H_diag=_MIN_H_DIAG,
        unscaled_grid=_MIN_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(1)
    E_solver = eval_diag(_MIN_W, s_solver, _MIN_H_DIAG, _MIN_GRID)

    assert s_solver.item() < 0
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"per_channel_diag W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}")


@dense_xfail
def test_per_group_wq_identity_first_negative_interval_optimal():
    w_bg = _MIN_W.reshape(1, 1, 2)
    s_solver = _piso_sweep_per_group(
        w_b=w_bg, H=_MIN_H, unscaled_grid=_MIN_GRID, eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(-1).squeeze(-1)
    E_solver = eval_dense(_MIN_W, s_solver, _MIN_H, _MIN_GRID)

    assert s_solver.item() < 0
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"per_group W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}")


def test_per_group_diag_wq_identity_first_negative_interval_optimal():
    w_bg = _MIN_W.reshape(1, 1, 2)
    s_solver = _piso_sweep_per_group_diag(
        w_b=w_bg,
        H_diag=_MIN_H_DIAG,
        unscaled_grid=_MIN_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(-1).squeeze(-1)
    E_solver = eval_diag(_MIN_W, s_solver, _MIN_H_DIAG, _MIN_GRID)

    assert s_solver.item() < 0
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"per_group_diag W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}")


@dense_xfail
def test_groups_sequential_wq_identity_first_negative_interval_optimal():
    w_bg = _MIN_W.reshape(1, 1, 2)
    s_solver = _piso_sweep_groups_sequential(
        w_b=w_bg, H=_MIN_H, unscaled_grid=_MIN_GRID, eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(-1).squeeze(-1)
    E_solver = eval_dense(_MIN_W, s_solver, _MIN_H, _MIN_GRID)

    assert s_solver.item() < 0
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"groups_sequential W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}")


@dense_xfail
def test_single_group_sequential_wq_identity_first_negative_interval_optimal():
    s_solver = _piso_sweep_single_group_sequential(
        w_orig=_MIN_W,
        w_updated=_MIN_W,
        H=_MIN_H,
        unscaled_grid=_MIN_GRID,
        group_size=2,
        group_idx=0,
        eps=SOLVER_EPS,
        allow_negative_s=True).squeeze(-1)
    E_solver = eval_dense(_MIN_W, s_solver, _MIN_H, _MIN_GRID)

    assert s_solver.item() < 0
    assert math.isclose(s_solver.item(), _MIN_A, abs_tol=1e-6)
    assert objective_leq(E_solver, torch.tensor([_MIN_E_OPT], dtype=DTYPE)), (
        f"single_group_sequential W=aQ: solver E={E_solver.item():.6f} > optimum E={_MIN_E_OPT:.6f}"
    )


# -----------------------------------------------------------------------------
# Known-optimum recovery: w = s_opt * Q for an on-grid Q and a known scale s_opt.
#
# If Q lies exactly on the grid, then q(w; s_opt) = round(s_opt Q / s_opt) = Q,
# so s = s_opt reconstructs w exactly. With H = G = I the objective is
#   E(s) = s^2 (q.q) - 2 s (q.w),  minimized at s = s_opt with E = -s_opt^2 (Q.Q),
# which is the global optimum (zero reconstruction error). Each solver receives
# H = I (or G = I) as a DENSE matrix for the dense/sequential variants and as a
# DIAGONAL vector for the *_diag variants; the recovered scale must equal s_opt.
#
# Uniqueness: on a symmetric grid (or without a decisive entry) (-s_opt, -Q) ties
# with (s_opt, Q), so the *sign* of the recovered scale is degenerate. To pin the
# scale exactly we include the asymmetric extreme grid[0] = -8 in Q: its mirror
# +8 is off-grid, so the sign flip is strictly worse and s = s_opt is the unique
# optimum. We test both positive and negative s_opt.
# -----------------------------------------------------------------------------
_REC_GRID = GRIDS["int4_signed"]  # [-8..7], asymmetric so -8 has no +8 mirror
_REC_GMIN = _REC_GRID[0]  # -8


def _identity_dense(D):
    return torch.eye(D, dtype=DTYPE)


def _identity_diag(D):
    return torch.ones(D, dtype=DTYPE)


# Each entry: name -> callable(w_bd, D, neg) -> recovered scale, shape [B].
# w_bd is [B, D]; group variants reshape it to a single group [B, 1, D].
def _run_per_channel(w_bd, D, neg):
    return _piso_sweep_per_channel(
        w_b=w_bd,
        H=_identity_dense(D),
        G=_identity_dense(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).squeeze(1)


def _run_per_channel_diag(w_bd, D, neg):
    return _piso_sweep_per_channel_diag(
        w_b=w_bd,
        H_diag=_identity_diag(D),
        G_diag=_identity_diag(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).squeeze(1)


def _run_per_group(w_bd, D, neg):
    return _piso_sweep_per_group(
        w_b=w_bd.reshape(-1, 1, D),
        H=_identity_dense(D),
        G=_identity_dense(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).reshape(-1)


def _run_per_group_diag(w_bd, D, neg):
    return _piso_sweep_per_group_diag(
        w_b=w_bd.reshape(-1, 1, D),
        H_diag=_identity_diag(D),
        G_diag=_identity_diag(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).reshape(-1)


def _run_groups_sequential(w_bd, D, neg):
    return _piso_sweep_groups_sequential(
        w_b=w_bd.reshape(-1, 1, D),
        H=_identity_dense(D),
        G=_identity_dense(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).reshape(-1)


def _run_single_group_sequential(w_bd, D, neg):
    return _piso_sweep_single_group_sequential(
        w_orig=w_bd,
        w_updated=w_bd,
        H=_identity_dense(D),
        G=_identity_dense(D),
        unscaled_grid=_REC_GRID,
        group_size=D,
        group_idx=0,
        eps=SOLVER_EPS,
        allow_negative_s=neg).reshape(-1)


RECOVERY_SOLVERS = {
    "per_channel": (_run_per_channel, "dense"),
    "per_channel_diag": (_run_per_channel_diag, "diag"),
    "per_group": (_run_per_group, "dense"),
    "per_group_diag": (_run_per_group_diag, "diag"),
    "groups_sequential": (_run_groups_sequential, "dense"),
    "single_group_sequential": (_run_single_group_sequential, "dense"),}


@pytest.mark.parametrize("solver_name", dense_params(RECOVERY_SOLVERS))
@pytest.mark.parametrize("s_opt", [0.3, -0.3, 1.5, -2.0, 0.05])
def test_recover_known_scale_from_on_grid_weights(solver_name, s_opt):
    """w = s_opt * Q with Q on the grid: the solver must recover s_opt exactly.

    Uniqueness of the scale (not just the objective) is enforced by seeding each
    row with grid[0] = -8, whose sign-flip +8 is off-grid.
    """
    run_solver, _ = RECOVERY_SOLVERS[solver_name]
    B, D = 6, 4
    neg = s_opt < 0

    g = torch.Generator().manual_seed(123)
    idx = torch.randint(0, _REC_GRID.numel(), (B, D), generator=g)
    Q = _REC_GRID[idx].to(DTYPE)
    # Seed a decisive extreme value per row so the sign of s_opt is uniquely optimal.
    Q[:, 0] = _REC_GMIN
    w = s_opt * Q

    s = run_solver(w, D, neg)

    assert torch.all(torch.isfinite(s))
    assert torch.allclose(
        s, torch.full_like(s, s_opt),
        atol=1e-6), (f"{solver_name} s_opt={s_opt}: recovered {s.tolist()} != {s_opt}")


@pytest.mark.parametrize("solver_name", dense_params(RECOVERY_SOLVERS))
@pytest.mark.parametrize("s_opt", [0.37, -0.21, 2.0, -1.3])
def test_on_grid_weights_reach_reconstruction_optimum(solver_name, s_opt):
    """Objective form of the above without the uniqueness seed.

    For random on-grid Q the sign of s_opt may be degenerate, but the solver must
    still reach the exact-reconstruction objective E(s_opt) = -s_opt^2 (Q.Q) (up
    to a small tolerance): it finds a scale at least as good as the planted one.
    """
    run_solver, _ = RECOVERY_SOLVERS[solver_name]
    B, D = 8, 4
    neg = s_opt < 0

    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, _REC_GRID.numel(), (B, D), generator=g)
    Q = _REC_GRID[idx].to(DTYPE)
    w = s_opt * Q

    s = run_solver(w, D, neg)

    E_solver = eval_dense(w, s, _identity_dense(D), _REC_GRID)
    E_opt = -(s_opt ** 2) * (Q * Q).sum(-1)
    assert objective_leq(
        E_solver,
        E_opt), (f"{solver_name} s_opt={s_opt}: max excess {(E_solver - E_opt).max().item():.3e}")


# -----------------------------------------------------------------------------
# Chunked PiSO (per-channel, positive scales only).
#
# The chunked solver is an exact reformulation of _piso_sweep_per_channel that
# only supports positive scales (allow_negative_s=False). We test it on both
# symmetric grids (where the model-level driver would request positive scales
# anyway) and asymmetric grids (where positive scales are a subset). For the
# grid-search comparison we always restrict the search to positive scales too, so
# the chunked positive-only optimum is compared against the positive-only best.
# Checks: (1) it reaches the positive-only grid-search optimum, (2) its objective
# matches the sequential per-channel solver run positive-only, (3) it recovers a
# planted positive scale, and (4) it rejects allow_negative_s=True.
# -----------------------------------------------------------------------------
SYMMETRIC_GRIDS = ["int4_narrow", "fp_e2m1", "fp_e3m2"]
ASYMMETRIC_GRIDS = ["int4_signed", "int3_signed"]


@dense_xfail
@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_per_channel_matches_positive_grid_search(grid_name, D, use_G):
    """Chunked solver reaches the POSITIVE-only grid-search optimum on any grid.

    The chunked solver only evaluates positive scales, so the reference grid
    search is also restricted to positive scales (allow_negative_s=False). This
    covers asymmetric grids too, where positive scales are a strict subset of the
    reachable ones.
    """
    grid = GRIDS[grid_name]
    B = 16
    w = random_weights(B, D, seed=1)
    H = spd_matrix(D, seed=2)
    G = spd_matrix(D, seed=3) if use_G else None

    s_solver = _chunked_piso_sweep_per_channel(
        w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False).squeeze(1)

    # The chunked solver only returns positive scales, regardless of grid symmetry.
    assert torch.all(
        s_solver > 0), (f"{grid_name} D={D} use_G={use_G}: non-positive scale {s_solver.tolist()}")

    def eval_fn(ww, ss, gg):
        return eval_dense(ww, ss, H, gg, G=G)

    E_solver = eval_dense(w, s_solver, H, grid, G=G)
    # Positive-only grid search (allow_negative_s=False restricts candidates to s>0).
    _, E_grid = grid_search_scales(w, eval_fn, grid, allow_negative_s=False)

    assert objective_leq(E_solver, E_grid), (
        f"{grid_name} D={D} use_G={use_G}: "
        f"max excess {(E_solver - E_grid).max().item():.3e}")


@dense_xfail
@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_matches_sequential_objective(grid_name, D, use_G):
    """Chunked and sequential per-channel solvers reach the same objective.

    Covers both symmetric and asymmetric grids: with allow_negative_s=False both
    solvers optimize strictly over positive scales, so their objectives must
    match on every grid. (This also guards the fix that stopped the sequential
    solver from committing a negative scale on the last, all-negative interval.)
    The scales may tie (several achieve the same optimal error), so we compare
    objectives, not scales.
    """
    grid = GRIDS[grid_name]
    B = 16
    w = random_weights(B, D, seed=4)
    H = spd_matrix(D, seed=5)
    G = spd_matrix(D, seed=6) if use_G else None

    s_seq = _piso_sweep_per_channel(
        w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False).squeeze(1)
    s_chunked = _chunked_piso_sweep_per_channel(
        w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False).squeeze(1)

    # allow_negative_s=False must never yield a negative scale, on any grid.
    assert torch.all(s_seq > 0), (
        f"{grid_name} D={D} use_G={use_G}: sequential returned non-positive scale "
        f"{s_seq[s_seq <= 0].tolist()}")

    E_seq = eval_dense(w, s_seq, H, grid, G=G)
    E_chunked = eval_dense(w, s_chunked, H, grid, G=G)
    assert torch.allclose(
        E_seq, E_chunked, atol=1e-8, rtol=0), (
            f"{grid_name} D={D} use_G={use_G}: "
            f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_diag_matches_sequential_objective(grid_name, D, use_G):
    """Chunked and sequential per-channel-diag solvers reach the same objective."""
    grid = GRIDS[grid_name]
    B = 16
    w = random_weights(B, D, seed=4)
    h_diag = spd_matrix(D, seed=5).diag().contiguous()
    g_diag = spd_matrix(D, seed=6).diag().contiguous() if use_G else None

    kw = dict(
        w_b=w,
        H_diag=h_diag,
        unscaled_grid=grid,
        G_diag=g_diag,
        eps=SOLVER_EPS,
        allow_negative_s=False)
    s_seq = _piso_sweep_per_channel_diag(**kw).squeeze(1)
    s_chunked = _chunked_piso_sweep_per_channel_diag(**kw).squeeze(1)

    assert torch.all(s_seq > 0) and torch.all(s_chunked > 0)
    E_seq = eval_diag(w, s_seq, h_diag, grid, g_diag=g_diag)
    E_chunked = eval_diag(w, s_chunked, h_diag, grid, g_diag=g_diag)
    assert torch.allclose(
        E_seq, E_chunked, atol=1e-8, rtol=0), (
            f"{grid_name} D={D} use_G={use_G}: "
            f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


# only the dense branch (solver_diag=False) reaches an unimplemented solver
@pytest.mark.parametrize("solver_diag", [pytest.param(False, marks=[dense_xfail]), True])
@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_group_matches_sequential_objective(
        solver_diag, grid_name, n_groups, group_size, use_G):
    """Chunked and sequential per-group solvers (dense + diag) reach the same objective."""
    grid = GRIDS[grid_name]
    B = 12
    D = n_groups * group_size
    w = random_weights(B, D, seed=11).reshape(B, n_groups, group_size)
    H = spd_matrix(D, seed=12)
    G = spd_matrix(D, seed=13) if use_G else None

    if solver_diag:
        h_diag = H.diag().contiguous()
        g_diag = G.diag().contiguous() if G is not None else None
        kw = dict(
            w_b=w,
            H_diag=h_diag,
            unscaled_grid=grid,
            G_diag=g_diag,
            eps=SOLVER_EPS,
            allow_negative_s=False)
        s_seq = _piso_sweep_per_group_diag(**kw).squeeze(-1)
        s_chunked = _chunked_piso_sweep_per_group_diag(**kw).squeeze(-1)
        H_eval, G_eval = torch.diag(h_diag), (torch.diag(g_diag) if g_diag is not None else None)
    else:
        kw = dict(w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False)
        s_seq = _piso_sweep_per_group(**kw).squeeze(-1)
        s_chunked = _chunked_piso_sweep_per_group(**kw).squeeze(-1)
        H_eval, G_eval = H, G

    E_seq = eval_group_dense(w, s_seq, H_eval, grid, group_size, G=G_eval)
    E_chunked = eval_group_dense(w, s_chunked, H_eval, grid, group_size, G=G_eval)
    assert torch.allclose(
        E_seq, E_chunked, atol=1e-8, rtol=0), (
            f"diag={solver_diag} {grid_name} ng={n_groups} gs={group_size} use_G={use_G}: "
            f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


def _eval_groups_sequential_obj(w_bg, s_bg, H, grid, group_size, G=None):
    """Full cross-group-aware objective E(sq) = sq^T H sq - 2 sq^T G w."""
    B, ng, gs = w_bg.shape
    G = H if G is None else G
    sq = (s_bg.unsqueeze(-1) * quantize_to_grid(w_bg, s_bg, grid)).reshape(B, ng * gs)
    w = w_bg.reshape(B, ng * gs)
    return (sq * (H @ sq.T).T).sum(-1) - 2 * (sq * (G @ w.T).T).sum(-1)


def _eval_single_group_obj(w_orig, w_upd, s, H, grid, gs, gi, G=None):
    """Per-group objective for group gi with the cross-group correction fixed."""
    B, D = w_orig.shape
    G = H if G is None else G
    gstart, gend = gi * gs, gi * gs + gs
    others = torch.cat((torch.arange(gstart), torch.arange(gend, D)))
    num = (G[gstart:gend, :] @ w_orig.T).T - (H[gstart:gend][:, others] @ w_upd[:, others].T).T
    sg = s.unsqueeze(-1) * quantize_to_grid(w_upd[:, gstart:gend], s, grid)
    Hgg = H[gstart:gend, gstart:gend]
    return (sg * (Hgg @ sg.T).T).sum(-1) - 2 * (sg * num).sum(-1)


@dense_xfail
@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS)
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_groups_sequential_matches_sequential_objective(
        grid_name, n_groups, group_size, use_G):
    """Chunked and sequential groups_sequential solvers reach the same objective.

    Symmetric grids only (chunked is positive-only). Scales may tie because the
    cross-group correction propagates the tie, so compare the full cross-group
    objective, not the raw scales.
    """
    grid = GRIDS[grid_name]
    B = 12
    D = n_groups * group_size
    w = random_weights(B, D, seed=11).reshape(B, n_groups, group_size)
    H = spd_matrix(D, seed=12)
    G = spd_matrix(D, seed=13) if use_G else None

    kw = dict(w_b=w, H=H, unscaled_grid=grid, G=G, eps=SOLVER_EPS, allow_negative_s=False)
    s_seq = _piso_sweep_groups_sequential(**kw).squeeze(-1)
    s_chunked = _chunked_piso_sweep_groups_sequential(**kw).squeeze(-1)

    E_seq = _eval_groups_sequential_obj(w, s_seq, H, grid, group_size, G=G)
    E_chunked = _eval_groups_sequential_obj(w, s_chunked, H, grid, group_size, G=G)
    assert torch.allclose(
        E_seq, E_chunked, atol=1e-8, rtol=0), (
            f"{grid_name} ng={n_groups} gs={group_size} use_G={use_G}: "
            f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


@dense_xfail
@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS)
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_single_group_sequential_matches_sequential_objective(
        grid_name, n_groups, group_size, use_G):
    """Chunked and sequential single_group_sequential solvers reach the same objective.

    Tests every group index with distinct float (w_orig) and error-corrected
    (w_updated) weights so the cross-group correction is exercised.
    """
    grid = GRIDS[grid_name]
    B = 12
    D = n_groups * group_size
    w_orig = random_weights(B, D, seed=21)
    w_upd = random_weights(B, D, seed=22)
    H = spd_matrix(D, seed=23)
    G = spd_matrix(D, seed=24) if use_G else None

    for gi in range(n_groups):
        kw = dict(
            w_orig=w_orig,
            w_updated=w_upd,
            H=H,
            unscaled_grid=grid,
            group_size=group_size,
            group_idx=gi,
            G=G,
            eps=SOLVER_EPS,
            allow_negative_s=False)
        s_seq = _piso_sweep_single_group_sequential(**kw).squeeze(-1)
        s_chunked = _chunked_piso_sweep_single_group_sequential(**kw).squeeze(-1)

        E_seq = _eval_single_group_obj(w_orig, w_upd, s_seq, H, grid, group_size, gi, G=G)
        E_chunked = _eval_single_group_obj(w_orig, w_upd, s_chunked, H, grid, group_size, gi, G=G)
        assert torch.allclose(
            E_seq, E_chunked, atol=1e-8, rtol=0), (
                f"{grid_name} ng={n_groups} gs={group_size} use_G={use_G} gi={gi}: "
                f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


@dense_xfail
@pytest.mark.parametrize("s_opt", [0.3, 1.5, 0.05])
def test_chunked_recover_known_positive_scale(s_opt):
    """w = s_opt * Q with Q on a symmetric grid: recover the positive scale.

    Uses a symmetric grid (allow_negative_s=False) since the chunked solver only
    supports positive scales. Seeding grid extremes makes s_opt uniquely optimal.
    """
    grid = GRIDS["int4_narrow"]  # symmetric [-7..7]
    B, D = 6, 4
    g = torch.Generator().manual_seed(123)
    idx = torch.randint(0, grid.numel(), (B, D), generator=g)
    Q = grid[idx].to(DTYPE)
    Q[:, 0] = grid[-1]  # a decisive extreme value per row
    w = s_opt * Q

    s = _chunked_piso_sweep_per_channel(
        w_b=w,
        H=_identity_dense(D),
        G=_identity_dense(D),
        unscaled_grid=grid,
        eps=SOLVER_EPS,
        allow_negative_s=False).squeeze(1)

    assert torch.all(torch.isfinite(s))
    assert torch.allclose(
        s, torch.full_like(s, s_opt),
        atol=1e-6), (f"chunked s_opt={s_opt}: recovered {s.tolist()} != {s_opt}")


@dense_xfail
def test_chunked_rejects_negative_scales():
    grid = GRIDS["int4_signed"]  # asymmetric -> would require allow_negative_s
    w = random_weights(4, 6, seed=1)
    H = spd_matrix(6, seed=2)
    with pytest.raises(ValueError, match="only supports positive scales"):
        _chunked_piso_sweep_per_channel(w_b=w, H=H, unscaled_grid=grid, allow_negative_s=True)


# -----------------------------------------------------------------------------
# Solver families: the registered chunked-piso family.
#
# The chunked family now implements every slot the 'piso' family does (per-channel
# and per-group, dense/diagonal, and the two sequential variants); all positive-
# scales only.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "is_group,use_diag,is_greedy,expected,arg_kind",
    [
        (False, False, False, _chunked_piso_sweep_per_channel, "dense"),
        (False, True, False, _chunked_piso_sweep_per_channel_diag, "diag"),
        (True, False, False, _chunked_piso_sweep_per_group, "dense"),
        (True, True, False, _chunked_piso_sweep_per_group_diag, "diag"),
        (True, False, True, _chunked_piso_sweep_groups_sequential, "greedy"),])
def test_chunked_piso_family_layer_dispatches(is_group, use_diag, is_greedy, expected, arg_kind):
    from brevitas.graph.piso.solvers import SCALE_SOLVER_REGISTRY
    fam = SCALE_SOLVER_REGISTRY.get("chunked-piso")
    solver, arg_kind_out = fam.get_layer_solver(is_group=is_group, use_diag=use_diag, is_greedy=is_greedy)
    assert solver is expected and arg_kind_out == arg_kind


@pytest.mark.parametrize(
    "use_diag,per_group_greedy,expected,tag",
    [
        (False, False, _chunked_piso_sweep_per_channel, "group-dense"),
        (True, False, _chunked_piso_sweep_per_channel_diag, "group-diag"),
        (False, True, _chunked_piso_sweep_single_group_sequential, "group-greedy"),])
def test_chunked_piso_family_group_dispatches(use_diag, per_group_greedy, expected, tag):
    from brevitas.graph.piso.solvers import SCALE_SOLVER_REGISTRY
    fam = SCALE_SOLVER_REGISTRY.get("chunked-piso")
    solver, tag_out = fam.get_group_solver(use_diag=use_diag, per_group_greedy=per_group_greedy)
    assert solver is expected and tag_out == tag


# =============================================================================
# Asymmetric quantizers: translated grids (integer zero point) and the additive
# zero-point term (free offset). See .agent_space/piso-gguf-extensions for the derivation.
# =============================================================================

UNSIGNED_GRIDS = {
    "uint4": int_grid(4, signed=False),
    "uint3": int_grid(3, signed=False),}


def random_zero_points(B, seed, scale=0.4, shape=None):
    """Arbitrary non-negative offsets.

    Only for tests that compare two code paths against each other; a zero point
    that does not cover the weight range is not a configuration the sweeps claim
    to solve (see calibrated_zero_point).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*(shape or (B, 1)), generator=g, dtype=DTYPE) * scale


def calibrated_zero_point(w):
    """z = -min(w, 0), the additive zero point a calibrated quantizer produces.

    The sweeps assume w + z >= 0. Otherwise the optimum can fall in the
    unbounded top interval, where every entry clamps to grid[0] and the
    reconstruction s*q - z = -z stops depending on the scale; the sweep only
    visits intervals delimited by transitions, so it would not find it.
    """
    return torch.clamp(-w.amin(-1, keepdim=True), min=0.)


def eval_affine_obj(w, s, z, H, G, grid):
    """E(s) = ||X w - X_tilde (s q - z)||^2 up to a constant, q on `grid`."""
    q = torch.clamp(torch.round((w + z) / s), grid[0], grid[-1])
    wh = s * q - z
    return (wh * (H @ wh.T).T).sum(-1) - 2 * (wh * (G @ w.T).T).sum(-1)


# -----------------------------------------------------------------------------
# translate_grid_by_zero_point
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("bit_width", [3, 4, 8])
def test_translate_grid_matches_brevitas_reconstruction(bit_width):
    """s * round_{G-z}(w/s) must equal Brevitas' asymmetric dequantization."""
    grid = int_grid(bit_width, signed=False)
    M = int(grid[-1].item())
    w = random_weights(6, 32, seed=71) * 0.3
    for z_val in [0, 1, M // 2, M]:
        z = torch.tensor(float(z_val), dtype=DTYPE)
        translated = translate_grid_by_zero_point(grid, z)
        assert bool((translated == grid - z).all())
        assert bool((translated == 0).any()), "translated grid must still contain 0"
        for s in [0.01, 0.05, 0.2]:
            # Brevitas: w_hat = s * (clamp(round(w/s + z), 0, M) - z)
            ref = s * (torch.clamp(torch.round(w / s + z), 0., float(M)) - z)
            ours = s * torch.clamp(torch.round(w / s), translated[0], translated[-1])
            assert torch.allclose(ref, ours), f"z={z_val} s={s}"


def test_translate_grid_rejects_invalid_inputs():
    grid = int_grid(4, signed=False)
    with pytest.raises(ValueError, match="integer-valued"):
        translate_grid_by_zero_point(grid, torch.tensor(1.5, dtype=DTYPE))
    with pytest.raises(ValueError, match="out of"):
        translate_grid_by_zero_point(grid, torch.tensor(16.0, dtype=DTYPE))
    with pytest.raises(ValueError, match="out of"):
        translate_grid_by_zero_point(grid, torch.tensor(-1.0, dtype=DTYPE))
    with pytest.raises(ValueError, match="grid\\[0\\] == 0"):
        translate_grid_by_zero_point(int_grid(4, signed=True), torch.tensor(0.0, dtype=DTYPE))


# -----------------------------------------------------------------------------
# Local (per-row) grids: a shared 1-D grid must equal the same grid tiled per row
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("variant", DENSE_DIAG_PARAMS)
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_per_channel_shared_grid_equals_tiled_local_grid(grid_name, variant):
    grid = UNSIGNED_GRIDS[grid_name]
    B, D = 6, 24
    w = random_weights(B, D, seed=72) * 0.3
    H, G = spd_matrix(D, seed=73), spd_matrix(D, seed=74)
    z = random_zero_points(B, seed=75)
    local = grid.unsqueeze(0).expand(B, -1).contiguous()
    fn, kw = ((_piso_sweep_per_channel, dict(H=H, G=G)) if variant == 'dense' else
              (_piso_sweep_per_channel_diag, dict(H_diag=torch.diag(H), G_diag=torch.diag(G))))
    shared = fn(
        w_b=w, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z, **kw)
    tiled = fn(
        w_b=w, unscaled_grid=local, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z, **kw)
    assert torch.equal(shared, tiled), variant


@pytest.mark.parametrize("variant", DENSE_DIAG_PARAMS)
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
@pytest.mark.parametrize("n_groups,group_size", [(2, 8), (3, 4)])
def test_per_group_shared_grid_equals_tiled_local_grid(grid_name, n_groups, group_size, variant):
    grid = UNSIGNED_GRIDS[grid_name]
    B, D = 5, n_groups * group_size
    w = (random_weights(B, D, seed=76) * 0.3).reshape(B, n_groups, group_size)
    H, G = spd_matrix(D, seed=77), spd_matrix(D, seed=78)
    z = random_zero_points(B, seed=79, shape=(B, n_groups, 1))
    local = grid.view(1, 1, -1).expand(B, n_groups, -1).contiguous()
    fn, kw = ((_piso_sweep_per_group, dict(H=H, G=G)) if variant == 'dense' else
              (_piso_sweep_per_group_diag, dict(H_diag=torch.diag(H), G_diag=torch.diag(G))))
    shared = fn(
        w_b=w, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z, **kw)
    tiled = fn(
        w_b=w, unscaled_grid=local, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z, **kw)
    assert torch.equal(shared, tiled), variant


@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_piso_grid_deltas_batched_matches_per_row(grid_name):
    """The batched grid-delta helper must agree row by row with the 1-D call."""
    grid = GRIDS[grid_name]
    zs = torch.tensor([[0.], [1.], [2.]], dtype=DTYPE)
    local = grid.unsqueeze(0) - zs
    _, kc_b, kn_b, d_b = _piso_grid_deltas(local)
    for i in range(local.shape[0]):
        _, kc, kn, d = _piso_grid_deltas(local[i])
        assert torch.equal(kc_b[i], kc) and torch.equal(kn_b[i], kn) and torch.equal(d_b[i], d)


# -----------------------------------------------------------------------------
# The zero point term: exact for an arbitrary G (not only the self-activation case)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
@pytest.mark.parametrize("use_diag", [pytest.param(False, marks=[dense_xfail]), True])
def test_offset_per_channel_matches_grid_search(grid_name, use_diag):
    grid = UNSIGNED_GRIDS[grid_name]
    B, D = 6, 24
    w = random_weights(B, D, seed=80) * 0.3
    H, G = spd_matrix(D, seed=81), spd_matrix(D, seed=82)
    z = calibrated_zero_point(w)
    if use_diag:
        H, G = torch.diag(torch.diag(H)), torch.diag(torch.diag(G))
        s = _piso_sweep_per_channel_diag(
            w_b=w,
            H_diag=torch.diag(H),
            G_diag=torch.diag(G),
            unscaled_grid=grid,
            eps=SOLVER_EPS,
            allow_negative_s=False,
            zero_point=z)
    else:
        s = _piso_sweep_per_channel(
            w_b=w,
            H=H,
            G=G,
            unscaled_grid=grid,
            eps=SOLVER_EPS,
            allow_negative_s=False,
            zero_point=z)
    E_solver = eval_affine_obj(w, s, z, H, G, grid)
    # grid-search reference over the same (positive) scale range
    s_max = ((w + z).abs().amax(-1, keepdim=True) / grid[-1]) * 4
    cand = torch.linspace(1e-4, 1.0, N_GRID_CANDIDATES, dtype=DTYPE).view(1, -1) * s_max
    E_grid = torch.stack(
        [eval_affine_obj(w, cand[:, j:j + 1], z, H, G, grid) for j in range(cand.shape[1])], -1)
    assert objective_leq(E_solver, E_grid.amin(-1))


@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
@pytest.mark.parametrize("n_groups,group_size", [(2, 8), (3, 4)])
@pytest.mark.parametrize("use_diag", [pytest.param(False, marks=[dense_xfail]), True])
def test_offset_per_group_matches_grid_search(grid_name, n_groups, group_size, use_diag):
    grid = UNSIGNED_GRIDS[grid_name]
    B, D = 5, n_groups * group_size
    w_flat = random_weights(B, D, seed=84) * 0.3
    w = w_flat.reshape(B, n_groups, group_size)
    H, G = spd_matrix(D, seed=85), spd_matrix(D, seed=86)
    z = calibrated_zero_point(w)
    if use_diag:
        H, G = torch.diag(torch.diag(H)), torch.diag(torch.diag(G))
        s = _piso_sweep_per_group_diag(
            w_b=w,
            H_diag=torch.diag(H),
            G_diag=torch.diag(G),
            unscaled_grid=grid,
            eps=SOLVER_EPS,
            allow_negative_s=False,
            zero_point=z).squeeze(-1)
    else:
        s = _piso_sweep_per_group(
            w_b=w,
            H=H,
            G=G,
            unscaled_grid=grid,
            eps=SOLVER_EPS,
            allow_negative_s=False,
            zero_point=z).squeeze(-1)
    # the independent-groups objective is separable, so check each group alone
    for k in range(n_groups):
        sl = slice(k * group_size, (k + 1) * group_size)
        Hk, Gk = H[sl, sl], G[sl, sl]
        E_solver = eval_affine_obj(w[:, k], s[:, k:k + 1], z[:, k], Hk, Gk, grid)
        s_max = ((w[:, k] + z[:, k]).abs().amax(-1, keepdim=True) / grid[-1]) * 4
        cand = torch.linspace(1e-4, 1.0, N_GRID_CANDIDATES, dtype=DTYPE).view(1, -1) * s_max
        E_grid = torch.stack([
            eval_affine_obj(w[:, k], cand[:, j:j + 1], z[:, k], Hk, Gk, grid)
            for j in range(cand.shape[1])],
                             -1)
        assert objective_leq(E_solver, E_grid.amin(-1)), f"group {k}"


OFFSET_NONE_SWEEPS = {
    'per_channel': 'dense',
    'per_channel_diag': 'diag',
    'per_group': 'dense',
    'per_group_diag': 'diag',
    'groups_sequential': 'dense',
    'single_group_sequential': 'dense',}


@pytest.mark.parametrize(
    "sweep",
    [
        pytest.param(n, marks=[dense_xfail] if k == 'dense' else []) for n,
        k in OFFSET_NONE_SWEEPS.items()])
@pytest.mark.parametrize("grid_name", list(GRIDS))
def test_offset_none_matches_zero_offset_for_every_sweep(grid_name, sweep):
    """zero_point=None and an all-zero zero_point must take identical code paths."""
    grid = GRIDS[grid_name]
    B, n_groups, group_size = 5, 3, 4
    D = n_groups * group_size
    w_flat = random_weights(B, D, seed=88) * 0.3
    w = w_flat.reshape(B, n_groups, group_size)
    w_upd = random_weights(B, D, seed=89) * 0.3
    H, G = spd_matrix(D, seed=90), spd_matrix(D, seed=91)
    hd, gd = torch.diag(H), torch.diag(G)
    zc = torch.zeros(B, 1, dtype=DTYPE)
    zg = torch.zeros(B, n_groups, 1, dtype=DTYPE)
    cases = {
        'per_channel': (_piso_sweep_per_channel, dict(w_b=w_flat, H=H, G=G), zc),
        'per_channel_diag':
            (_piso_sweep_per_channel_diag, dict(w_b=w_flat, H_diag=hd, G_diag=gd), zc),
        'per_group': (_piso_sweep_per_group, dict(w_b=w, H=H, G=G), zg),
        'per_group_diag': (_piso_sweep_per_group_diag, dict(w_b=w, H_diag=hd, G_diag=gd), zg),
        'groups_sequential': (_piso_sweep_groups_sequential, dict(w_b=w, H=H, G=G), zg),
        'single_group_sequential': (
            _piso_sweep_single_group_sequential,
            dict(w_orig=w_flat, w_updated=w_upd, H=H, G=G, group_size=group_size, group_idx=1),
            zc),}
    fn, kw, z0 = cases[sweep]
    base = fn(unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, **kw)
    zeroed = fn(unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z0, **kw)
    assert torch.equal(base, zeroed), sweep


def test_offset_rejects_negative_scales():
    grid = UNSIGNED_GRIDS["uint4"]
    B, D = 4, 12
    w = random_weights(B, D, seed=92) * 0.3
    H = spd_matrix(D, seed=93)
    z = random_zero_points(B, seed=94)
    with pytest.raises(ValueError, match="positive scales"):
        _piso_sweep_per_channel_diag(
            w_b=w,
            H_diag=torch.diag(H),
            unscaled_grid=grid,
            eps=SOLVER_EPS,
            allow_negative_s=True,
            zero_point=z)


@dense_xfail
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_groups_sequential_commits_dequantized_with_offset(grid_name):
    """A processed group must commit s*q - z, not s*q, so later groups subtract
    the reconstruction the quantizer actually produces."""
    grid = UNSIGNED_GRIDS[grid_name]
    B, n_groups, group_size = 4, 3, 4
    D = n_groups * group_size
    w = (random_weights(B, D, seed=95) * 0.3).reshape(B, n_groups, group_size)
    H, G = spd_matrix(D, seed=96), spd_matrix(D, seed=97)
    z = calibrated_zero_point(w)
    s = _piso_sweep_groups_sequential(
        w_b=w, H=H, G=G, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False,
        zero_point=z).squeeze(-1)
    # Group 0 has no committed prefix, so it must match the independent sweep.
    s_ind = _piso_sweep_per_group(
        w_b=w, H=H, G=G, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False,
        zero_point=z).squeeze(-1)
    assert torch.allclose(s[:, 0], s_ind[:, 0])

    # The full objective must not be worse than the absmax-style baseline.
    def full_obj(scales):
        q = torch.clamp(torch.round((w + z) / scales.unsqueeze(-1)), grid[0], grid[-1])
        sq = (scales.unsqueeze(-1) * q - z).reshape(B, D)
        wf = w.reshape(B, D)
        return (sq * (H @ sq.T).T).sum(-1) - 2 * (sq * (G @ wf.T).T).sum(-1)

    baseline = (w + z).abs().amax(-1) / grid[-1]
    assert bool(torch.all(full_obj(s) <= full_obj(baseline) + ATOL))


@dense_xfail
@pytest.mark.parametrize("grid_name", list(GRIDS))
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
def test_sequential_shared_grid_equals_tiled_local_grid(grid_name, n_groups, group_size):
    """The sequential sweeps take a per-group translated grid too, which is what
    lets the greedy ordering and the group-interleaved updater run with an
    integer zero point."""
    grid = GRIDS[grid_name]
    B, D = 5, n_groups * group_size
    w_flat = random_weights(B, D, seed=99) * 0.3
    w = w_flat.reshape(B, n_groups, group_size)
    w_upd = random_weights(B, D, seed=100) * 0.3
    H, G = spd_matrix(D, seed=101), spd_matrix(D, seed=102)

    # groups_sequential: one grid row per (channel, group)
    local_g = grid.view(1, 1, -1).expand(B, n_groups, -1).contiguous()
    shared = _piso_sweep_groups_sequential(
        w_b=w, H=H, G=G, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False)
    tiled = _piso_sweep_groups_sequential(
        w_b=w, H=H, G=G, unscaled_grid=local_g, eps=SOLVER_EPS, allow_negative_s=False)
    assert torch.equal(shared, tiled), "groups_sequential"

    # single_group_sequential: one grid row per channel, for the group in hand
    local_c = grid.view(1, -1).expand(B, -1).contiguous()
    for group_idx in range(n_groups):
        kw = dict(
            w_orig=w_flat,
            w_updated=w_upd,
            H=H,
            G=G,
            group_size=group_size,
            group_idx=group_idx,
            eps=SOLVER_EPS,
            allow_negative_s=False)
        shared = _piso_sweep_single_group_sequential(unscaled_grid=grid, **kw)
        tiled = _piso_sweep_single_group_sequential(unscaled_grid=local_c, **kw)
        assert torch.equal(shared, tiled), f"single_group_sequential group {group_idx}"


# =============================================================================
# The chunked family implements the asymmetric extensions too: it must agree
# with the reference sweeps on both a free offset and a translated grid.
# =============================================================================

CHUNKED_PAIRS = [
    ("per_channel", _piso_sweep_per_channel, _chunked_piso_sweep_per_channel, 'channel', 'dense'),
    (
        "per_channel_diag",
        _piso_sweep_per_channel_diag,
        _chunked_piso_sweep_per_channel_diag,
        'channel',
        'diag'),
    ("per_group", _piso_sweep_per_group, _chunked_piso_sweep_per_group, 'group', 'dense'),
    (
        "per_group_diag",
        _piso_sweep_per_group_diag,
        _chunked_piso_sweep_per_group_diag,
        'group',
        'diag'),
    (
        "groups_sequential",
        _piso_sweep_groups_sequential,
        _chunked_piso_sweep_groups_sequential,
        'group',
        'dense'),]


def chunked_pair_params(pairs):
    """pytest params for CHUNKED_PAIRS, xfailing the entries backed by a dense slot."""
    return [
        pytest.param(*p, marks=[dense_xfail] if p[4] == 'dense' else [], id=p[0]) for p in pairs]


def _asym_pair_inputs(layout, hessian, n_groups, group_size, grid):
    B, D = 5, n_groups * group_size
    w_flat = random_weights(B, D, seed=411) * 0.3
    H, G = spd_matrix(D, seed=412), spd_matrix(D, seed=413)
    kw = (dict(H_diag=torch.diag(H), G_diag=torch.diag(G)) if hessian == 'diag' else dict(H=H, G=G))
    if layout == 'group':
        w = w_flat.reshape(B, n_groups, group_size)
        return w, kw, calibrated_zero_point(w)
    return w_flat, kw, calibrated_zero_point(w_flat)


@pytest.mark.parametrize(
    "name,ref_fn,chunked_fn,layout,hessian", chunked_pair_params(CHUNKED_PAIRS))
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_chunked_matches_reference_with_free_offset(
        name, ref_fn, chunked_fn, layout, hessian, grid_name):
    grid = UNSIGNED_GRIDS[grid_name]
    w, kw, z = _asym_pair_inputs(layout, hessian, 3, 8, grid)
    call = dict(
        w_b=w, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z, **kw)
    assert torch.allclose(ref_fn(**call), chunked_fn(**call), rtol=1e-6, atol=1e-9), name


@pytest.mark.parametrize(
    "name,ref_fn,chunked_fn,layout,hessian", chunked_pair_params(CHUNKED_PAIRS))
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_chunked_matches_reference_with_translated_grid(
        name, ref_fn, chunked_fn, layout, hessian, grid_name):
    """An integer zero point gives every row (and group) its own grid."""
    grid = UNSIGNED_GRIDS[grid_name]
    n_groups, group_size = 3, 8
    w, kw, _ = _asym_pair_inputs(layout, hessian, n_groups, group_size, grid)
    B = w.shape[0]
    shape = (B, n_groups, 1) if layout == 'group' else (B, 1)
    g = torch.Generator().manual_seed(414)
    zeta = torch.randint(0, int(grid[-1].item()) + 1, shape, generator=g).to(DTYPE)
    local_grid = translate_grid_by_zero_point(grid, zeta)
    call = dict(w_b=w, unscaled_grid=local_grid, eps=SOLVER_EPS, allow_negative_s=False, **kw)
    assert torch.allclose(ref_fn(**call), chunked_fn(**call), rtol=1e-6, atol=1e-9), name


@dense_xfail
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_chunked_single_group_sequential_asymmetric(grid_name):
    grid = UNSIGNED_GRIDS[grid_name]
    B, n_groups, group_size = 5, 3, 8
    D = n_groups * group_size
    w_orig = random_weights(B, D, seed=415) * 0.3
    w_upd = random_weights(B, D, seed=416) * 0.3
    H, G = spd_matrix(D, seed=417), spd_matrix(D, seed=418)
    group_idx = 1
    sl = slice(group_idx * group_size, (group_idx + 1) * group_size)
    base = dict(
        w_orig=w_orig,
        w_updated=w_upd,
        H=H,
        G=G,
        group_size=group_size,
        group_idx=group_idx,
        eps=SOLVER_EPS,
        allow_negative_s=False)
    # free offset
    call = dict(unscaled_grid=grid, zero_point=calibrated_zero_point(w_upd[:, sl]), **base)
    assert torch.allclose(
        _piso_sweep_single_group_sequential(**call),
        _chunked_piso_sweep_single_group_sequential(**call),
        rtol=1e-6,
        atol=1e-9)
    # translated grid
    g = torch.Generator().manual_seed(419)
    zeta = torch.randint(0, int(grid[-1].item()) + 1, (B, 1), generator=g).to(DTYPE)
    call = dict(unscaled_grid=translate_grid_by_zero_point(grid, zeta), **base)
    assert torch.allclose(
        _piso_sweep_single_group_sequential(**call),
        _chunked_piso_sweep_single_group_sequential(**call),
        rtol=1e-6,
        atol=1e-9)


@dense_xfail
@pytest.mark.parametrize("grid_name", ["uint4", "uint3"])
def test_chunked_handles_rows_with_no_negative_transition(grid_name):
    """An unsigned grid with a covering zero point makes every transition
    positive, which the j_min skip must not overrun."""
    grid = UNSIGNED_GRIDS[grid_name]
    B, n_groups, group_size = 4, 2, 8
    D = n_groups * group_size
    w = (random_weights(B, D, seed=420) * 0.3).reshape(B, n_groups, group_size)
    z = calibrated_zero_point(w)
    assert bool(torch.all(w + z >= 0.)), "precondition: the zero point covers the range"
    H, G = spd_matrix(D, seed=421), spd_matrix(D, seed=422)
    call = dict(
        w_b=w, H=H, G=G, unscaled_grid=grid, eps=SOLVER_EPS, allow_negative_s=False, zero_point=z)
    assert torch.allclose(
        _piso_sweep_groups_sequential(**call),
        _chunked_piso_sweep_groups_sequential(**call),
        rtol=1e-6,
        atol=1e-9)


# =============================================================================
# Hierarchical scales: the macro problem is the base problem one level up. The
# llama.cpp-style reduction uses the plain per-group sum of diag(H) as curvature
# and keeps the solved scales as the weights, so it is solved by the *existing*
# diagonal group sweep.
# =============================================================================
MACRO_B, MACRO_K, MACRO_GS, MACRO_NSB = 3, 8, 4, 2
MACRO_D = MACRO_K * MACRO_GS


def _macro_inputs(seed=11):
    """A solved group-scale problem to reduce: weights, scales, grids, H."""
    w = (random_weights(MACRO_B, MACRO_D, seed=seed) * 0.3).reshape(MACRO_B, MACRO_K, MACRO_GS)
    code_grid = int_grid(6, signed=False)
    H = spd_matrix(MACRO_D, seed=seed + 1)
    optimal_s = (w.abs().amax(-1) / int_grid(4)[-1]).clamp_min(1e-6)
    return w, optimal_s, code_grid, H


@pytest.mark.parametrize("pass_diag", [False, True])
def test_macro_problem_reduces_to_the_group_diagonal_sum(pass_diag):
    """H_diag[j] is the plain sum of diag(H) over group j, whether H arrives dense
    or already as its diagonal; scales are the solved scales, unchanged."""
    w, optimal_s, _, H = _macro_inputs()
    H_in = torch.diag(H) if pass_diag else H
    scales, H_diag = build_quant_scale_problem(
        w_b=w, optimal_s=optimal_s, H=H_in, n_macro_blocks=MACRO_NSB, use_diag=pass_diag)

    expected = torch.diag(H).reshape(MACRO_K, MACRO_GS).sum(-1)  # [K]
    assert torch.allclose(H_diag, expected.repeat(MACRO_B))
    assert torch.equal(scales, optimal_s.reshape(MACRO_B, MACRO_NSB, MACRO_K // MACRO_NSB))


def test_macro_curvature_ignores_the_codes():
    """Unlike the exact q^T H q curvature, the sum-of-diagonals one does not depend
    on the weights (hence on the frozen assignment) at all."""
    w, optimal_s, _, H = _macro_inputs()
    _, H_diag_a = build_quant_scale_problem(
        w_b=w, optimal_s=optimal_s, H=H, n_macro_blocks=MACRO_NSB, use_diag=False)
    _, H_diag_b = build_quant_scale_problem(
        w_b=w * 7.0, optimal_s=optimal_s * 3.0, H=H, n_macro_blocks=MACRO_NSB, use_diag=False)
    assert torch.equal(H_diag_a, H_diag_b)


def _macro_objective(scales, d, H_diag, code_grid):
    """The reduced macro objective sum_j H_diag[j] (s_j - d c_j)^2 per super-block.

    scales and H_diag are [B, NSB, per_super_block]; d is [B, NSB].
    """
    c = torch.clamp(torch.round(scales / d.unsqueeze(-1)), code_grid[0], code_grid[-1])
    return (H_diag * (c * d.unsqueeze(-1) - scales) ** 2).sum(-1)


def test_macro_sweep_beats_a_grid_search_over_d():
    """The reduction plus the diagonal sweep must reach the best macro scale on the
    reduced (sum-of-diagonals) objective, checked against a dense grid search."""
    w, optimal_s, code_grid, H = _macro_inputs()
    scales, H_diag = build_quant_scale_problem(
        w_b=w, optimal_s=optimal_s, H=H, n_macro_blocks=MACRO_NSB, use_diag=False)
    per_super_block = MACRO_K // MACRO_NSB
    d = _piso_sweep_per_group_diag(
        w_b=scales.reshape(1, MACRO_B * MACRO_NSB, per_super_block),
        H_diag=H_diag,
        unscaled_grid=code_grid,
        eps=SOLVER_EPS,
        allow_negative_s=False).reshape(MACRO_B, MACRO_NSB)

    H_diag_bk = H_diag.reshape(MACRO_B, MACRO_NSB, per_super_block)
    err = lambda dd: _macro_objective(scales, dd, H_diag_bk, code_grid)
    cand = torch.linspace(1e-6, 1.0, 4001, dtype=DTYPE)
    hi = scales.amax(-1) / code_grid[-1] * 4  # [B, NSB]
    E_grid = torch.stack([err(c * hi) for c in cand], -1).amin(-1)
    assert objective_leq(err(d), E_grid)


def test_macro_channel_folded_into_group_axis_equals_per_channel_loop():
    """The macro curvature is per channel; folding the channel into the group axis
    (one diagonal spanning all channels) must equal a per-channel loop."""
    code_grid = int_grid(4, signed=False)
    gen = torch.Generator().manual_seed(314)
    scales = torch.rand(MACRO_B, MACRO_K, generator=gen, dtype=DTYPE) * 0.02 + 0.001
    beta = torch.rand(MACRO_B, MACRO_K, generator=gen, dtype=DTYPE) + 0.5
    gs = MACRO_K // MACRO_NSB
    call = lambda s, h: _piso_sweep_per_group_diag(
        w_b=s, H_diag=h, unscaled_grid=code_grid, eps=SOLVER_EPS, allow_negative_s=False)
    loop = torch.stack([
        call(scales[b].reshape(1, MACRO_NSB, gs), beta[b]).reshape(MACRO_NSB)
        for b in range(MACRO_B)])
    folded = call(scales.reshape(1, MACRO_B * MACRO_NSB, gs),
                  beta.reshape(-1)).reshape(MACRO_B, MACRO_NSB)
    assert torch.equal(loop, folded)
