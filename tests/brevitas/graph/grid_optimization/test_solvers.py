# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PiSO interval-sweep solvers (brevitas.graph.grid_optimization.solvers).

Each solver returns, in closed form, the weight-quantization scale that minimizes
the reconstruction objective

    E(s) = s^2 * (q^T H q) - 2 s * (q^T b),     q = clamp(round(w / s), grid)

with a diagonal H and the linear term b = G w (b = H w by default). This is the
scale-dependent part of ||X w - s X_tilde q||^2 up to the s-independent constant
w^T X^T X w.

These tests evaluate that objective directly and compare each solver's result
against a brute-force grid search over candidate scales. Because PiSO is exact,
the solver's objective must be no larger than the best objective found by grid
search (up to a small tolerance).

Solver-agnostic tests are parametrized over the sequential 'piso' and the
'chunked-piso' reformulation; the chunked solver is skipped on cases that require
negative scales, since it only supports positive scales.
"""
import math

import pytest
import torch

from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers.chunked_piso import \
    _chunked_piso_sweep_per_channel_diag
from brevitas.graph.grid_optimization.solvers.chunked_piso import _chunked_piso_sweep_per_group_diag
from brevitas.graph.grid_optimization.solvers.piso import _piso_sweep_per_channel_diag
from brevitas.graph.grid_optimization.solvers.piso import _piso_sweep_per_group_diag
from brevitas.graph.grid_optimization.utils import make_fp_grid
from brevitas.graph.grid_optimization.utils import make_int_grid

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


# -----------------------------------------------------------------------------
# Diagonal solvers under a common interface:
# (w_b, H_diag, unscaled_grid, b, eps, allow_negative_s).
# -----------------------------------------------------------------------------
PER_CHANNEL_DIAG_SOLVERS = {
    "piso": _piso_sweep_per_channel_diag,
    "chunked-piso": _chunked_piso_sweep_per_channel_diag,}

PER_GROUP_DIAG_SOLVERS = {
    "piso": _piso_sweep_per_group_diag,
    "chunked-piso": _chunked_piso_sweep_per_group_diag,}


def skip_if_unsupported(solver_name, allow_negative_s):
    """Skip the chunked solver on cases that need negative scales."""
    if solver_name == "chunked-piso" and allow_negative_s:
        pytest.skip("chunked-piso only supports positive scales (allow_negative_s=False)")


def linear_term(w, g_diag):
    """b = diag(G) * w, or None for the self-activation objective."""
    if g_diag is None:
        return None
    return (g_diag * w.reshape(w.shape[0], -1)).reshape(w.shape)


# -----------------------------------------------------------------------------
# Tests: per-channel.
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
        b=linear_term(w, g_diag),
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
# Regression: allow_negative_s=False must reject negative scales even on
# asymmetric grids. The sweep enters a fully-negative region as s -> 0; on the
# last step (where every row's interval is negative) the closed-form minimizer
# can be negative, and it must NOT be committed.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("grid_name", ["int4_signed", "int3_signed"])
@pytest.mark.parametrize("D", [4, 8, 9])
@pytest.mark.parametrize("use_G", [False, True])
def test_positive_only_rejects_negative_scale_on_asymmetric_grid(grid_name, D, use_G):
    grid = GRIDS[grid_name]
    assert allows_negative(grid)  # sanity: these grids are asymmetric
    B = 16
    # Sweep several seeds: the negative-scale-on-last-interval path only fires
    # for some weight/covariance draws, so a single fixed seed is a weak guard.
    for seed in range(16):
        w = random_weights(B, D, seed=seed)
        h_diag = spd_matrix(D, seed=seed + 100).diag().contiguous()
        g_diag = spd_matrix(D, seed=seed + 200).diag().contiguous() if use_G else None
        s = _piso_sweep_per_channel_diag(
            w_b=w,
            H_diag=h_diag,
            unscaled_grid=grid,
            b=linear_term(w, g_diag),
            eps=SOLVER_EPS,
            allow_negative_s=False).squeeze(1)
        assert torch.all(torch.isfinite(s))
        assert torch.all(s > 0), (
            f"{grid_name} D={D} use_G={use_G} seed={seed}: "
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
_MIN_H_DIAG = torch.ones(2, dtype=DTYPE)
# Exact optimum: s = a reconstructs w, E = a^2 (Q.Q) - 2 a (a Q.Q) = -a^2 (Q.Q).
_MIN_E_OPT = -(_MIN_A ** 2) * (_MIN_Q @ _MIN_Q)


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


# -----------------------------------------------------------------------------
# Known-optimum recovery: w = s_opt * Q for an on-grid Q and a known scale s_opt.
#
# If Q lies exactly on the grid, then q(w; s_opt) = round(s_opt Q / s_opt) = Q,
# so s = s_opt reconstructs w exactly. With H = G = I the objective is
#   E(s) = s^2 (q.q) - 2 s (q.w),  minimized at s = s_opt with E = -s_opt^2 (Q.Q),
# which is the global optimum (zero reconstruction error).
#
# Uniqueness: on a symmetric grid (or without a decisive entry) (-s_opt, -Q) ties
# with (s_opt, Q), so the *sign* of the recovered scale is degenerate. To pin the
# scale exactly we include the asymmetric extreme grid[0] = -8 in Q: its mirror
# +8 is off-grid, so the sign flip is strictly worse and s = s_opt is the unique
# optimum. We test both positive and negative s_opt.
# -----------------------------------------------------------------------------
_REC_GRID = GRIDS["int4_signed"]  # [-8..7], asymmetric so -8 has no +8 mirror
_REC_GMIN = _REC_GRID[0]  # -8


def _identity_diag(D):
    return torch.ones(D, dtype=DTYPE)


# Each entry: name -> callable(w_bd, D, neg) -> recovered scale, shape [B].
# w_bd is [B, D]; the group variant reshapes it to a single group [B, 1, D].
def _run_per_channel_diag(w_bd, D, neg):
    return _piso_sweep_per_channel_diag(
        w_b=w_bd,
        H_diag=_identity_diag(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).squeeze(1)


def _run_per_group_diag(w_bd, D, neg):
    return _piso_sweep_per_group_diag(
        w_b=w_bd.reshape(-1, 1, D),
        H_diag=_identity_diag(D),
        unscaled_grid=_REC_GRID,
        eps=SOLVER_EPS,
        allow_negative_s=neg).reshape(-1)


RECOVERY_SOLVERS = {
    "per_channel_diag": _run_per_channel_diag,
    "per_group_diag": _run_per_group_diag,}


@pytest.mark.parametrize("solver_name", list(RECOVERY_SOLVERS))
@pytest.mark.parametrize("s_opt", [0.3, -0.3, 1.5, -2.0, 0.05])
def test_recover_known_scale_from_on_grid_weights(solver_name, s_opt):
    """w = s_opt * Q with Q on the grid: the solver must recover s_opt exactly.

    Uniqueness of the scale (not just the objective) is enforced by seeding each
    row with grid[0] = -8, whose sign-flip +8 is off-grid.
    """
    run_solver = RECOVERY_SOLVERS[solver_name]
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


@pytest.mark.parametrize("solver_name", list(RECOVERY_SOLVERS))
@pytest.mark.parametrize("s_opt", [0.37, -0.21, 2.0, -1.3])
def test_on_grid_weights_reach_reconstruction_optimum(solver_name, s_opt):
    """Objective form of the above without the uniqueness seed.

    For random on-grid Q the sign of s_opt may be degenerate, but the solver must
    still reach the exact-reconstruction objective E(s_opt) = -s_opt^2 (Q.Q) (up
    to a small tolerance): it finds a scale at least as good as the planted one.
    """
    run_solver = RECOVERY_SOLVERS[solver_name]
    B, D = 8, 4
    neg = s_opt < 0

    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, _REC_GRID.numel(), (B, D), generator=g)
    Q = _REC_GRID[idx].to(DTYPE)
    w = s_opt * Q

    s = run_solver(w, D, neg)

    E_solver = eval_diag(w, s, _identity_diag(D), _REC_GRID)
    E_opt = -(s_opt ** 2) * (Q * Q).sum(-1)
    assert objective_leq(
        E_solver,
        E_opt), (f"{solver_name} s_opt={s_opt}: max excess {(E_solver - E_opt).max().item():.3e}")


# -----------------------------------------------------------------------------
# Chunked PiSO (positive scales only).
#
# The chunked solver is an exact reformulation of the PiSO sweep that only
# supports positive scales (allow_negative_s=False). We test it on both symmetric
# grids and asymmetric grids (where positive scales are a subset). Checks: (1) its
# objective matches the sequential solver run positive-only, (2) it recovers a
# planted positive scale, and (3) it rejects allow_negative_s=True.
# -----------------------------------------------------------------------------
SYMMETRIC_GRIDS = ["int4_narrow", "fp_e2m1", "fp_e3m2"]
ASYMMETRIC_GRIDS = ["int4_signed", "int3_signed"]


@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("D", [4, 8])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_diag_matches_sequential_objective(grid_name, D, use_G):
    """Chunked and sequential per-channel solvers reach the same objective."""
    grid = GRIDS[grid_name]
    B = 16
    w = random_weights(B, D, seed=4)
    h_diag = spd_matrix(D, seed=5).diag().contiguous()
    g_diag = spd_matrix(D, seed=6).diag().contiguous() if use_G else None

    kw = dict(
        w_b=w,
        H_diag=h_diag,
        unscaled_grid=grid,
        b=linear_term(w, g_diag),
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


@pytest.mark.parametrize("grid_name", SYMMETRIC_GRIDS + ASYMMETRIC_GRIDS)
@pytest.mark.parametrize("n_groups,group_size", [(2, 4), (3, 2)])
@pytest.mark.parametrize("use_G", [False, True])
def test_chunked_group_matches_sequential_objective(grid_name, n_groups, group_size, use_G):
    """Chunked and sequential per-group solvers reach the same objective."""
    grid = GRIDS[grid_name]
    B = 12
    D = n_groups * group_size
    w = random_weights(B, D, seed=11).reshape(B, n_groups, group_size)
    h_diag = spd_matrix(D, seed=12).diag().contiguous()
    g_diag = spd_matrix(D, seed=13).diag().contiguous() if use_G else None
    kw = dict(
        w_b=w,
        H_diag=h_diag,
        unscaled_grid=grid,
        b=linear_term(w, g_diag),
        eps=SOLVER_EPS,
        allow_negative_s=False)
    s_seq = _piso_sweep_per_group_diag(**kw).squeeze(-1)
    s_chunked = _chunked_piso_sweep_per_group_diag(**kw).squeeze(-1)

    H_eval = torch.diag(h_diag)
    G_eval = None if g_diag is None else torch.diag(g_diag)
    E_seq = eval_group_dense(w, s_seq, H_eval, grid, group_size, G=G_eval)
    E_chunked = eval_group_dense(w, s_chunked, H_eval, grid, group_size, G=G_eval)
    assert torch.allclose(
        E_seq, E_chunked, atol=1e-8, rtol=0), (
            f"{grid_name} ng={n_groups} gs={group_size} use_G={use_G}: "
            f"max |dE| {(E_seq - E_chunked).abs().max().item():.3e}")


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

    s = _chunked_piso_sweep_per_channel_diag(
        w_b=w, H_diag=_identity_diag(D), unscaled_grid=grid, eps=SOLVER_EPS,
        allow_negative_s=False).squeeze(1)

    assert torch.all(torch.isfinite(s))
    assert torch.allclose(
        s, torch.full_like(s, s_opt),
        atol=1e-6), (f"chunked s_opt={s_opt}: recovered {s.tolist()} != {s_opt}")


def test_chunked_rejects_negative_scales():
    grid = GRIDS["int4_signed"]  # asymmetric -> would require allow_negative_s
    w = random_weights(4, 6, seed=1)
    with pytest.raises(ValueError, match="only supports positive scales"):
        _chunked_piso_sweep_per_channel_diag(
            w_b=w, H_diag=_identity_diag(6), unscaled_grid=grid, allow_negative_s=True)


# -----------------------------------------------------------------------------
# The registered families: solve hands a single group per row to the per-channel
# sweep and several to the per-group one.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,per_channel,per_group",
    [
        ("piso", _piso_sweep_per_channel_diag, _piso_sweep_per_group_diag),
        ("chunked-piso", _chunked_piso_sweep_per_channel_diag, _chunked_piso_sweep_per_group_diag),
    ])
@pytest.mark.parametrize("n_groups", [1, 3])
def test_family_solve_dispatches_on_the_number_of_groups(name, per_channel, per_group, n_groups):
    grid = GRIDS["int4_narrow"]
    B, group_size = 5, 4
    D = n_groups * group_size
    w = random_weights(B, D, seed=3).reshape(B, n_groups, group_size)
    h_diag = spd_matrix(D, seed=4).diag().contiguous()
    g_diag = spd_matrix(D, seed=5).diag().contiguous()
    b = linear_term(w, g_diag)
    s = SCALE_SOLVER_REGISTRY.get(name).solve(
        w, b, h_diag, grid, allow_negative_s=False, with_offset=False, eps=SOLVER_EPS)
    if n_groups == 1:
        expected = per_channel(w[:, 0], h_diag, grid, b[:, 0], SOLVER_EPS, False)
    else:
        expected = per_group(w, h_diag, grid, b, SOLVER_EPS, False)
    assert torch.equal(s, expected)
