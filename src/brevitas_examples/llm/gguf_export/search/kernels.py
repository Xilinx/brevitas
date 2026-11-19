# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This code is adapted from https://github.com/ggml-org/llama.cpp, under the following LICENSE:
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""llama.cpp's block-level scale searches (ggml/src/ggml-quants.c), vectorised and
generalised to any integer code range [code_min, code_max].

Every kernel fits each block along the last dimension of its input independently,
minimising the weighted objective sum(quad * r^2) - 2 * sum(lin * r) over the
reconstruction r of the block. With lin = quad * x, that is the weighted squared
error sum(quad * (x - r)^2) llama.cpp minimises, up to a constant. A block without
an offset is reconstructed as scale * code, one with an offset as scale * code +
offset with codes counted from the lowest one, i.e. in [0, code_max - code_min].
A positive placement, which puts non-negative values such as the per-block scales
of a block on a grid of their own, reconstructs step * code with codes in [0, nmax].
"""

from typing import Callable
from typing import Tuple

import torch

__all__ = [
    "affine_scale_search",
    "affine_seed",
    "positive_absmax_grid",
    "positive_grid_search",
    "signed_absmax_grid",
    "symmetric_descent_search",
    "symmetric_scale_search",
    "symmetric_seed",]

# A block whose largest magnitude is below this is all-zero (ggml's GROUP_MAX_EPS).
_ZERO_EPS = 1e-15


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """numerator / denominator, and zero where the denominator is not positive."""
    positive = denominator > 0
    return torch.where(
        positive,
        numerator / torch.where(positive, denominator, torch.ones_like(denominator)),
        torch.zeros_like(numerator))


def _seed(x: torch.Tensor, code_min: int, code_max: int,
          positive_scale: bool) -> Tuple[Callable[[float], torch.Tensor], torch.Tensor]:
    """The inverse scale of the absmax seed, as a function of a trial offset, and which
    blocks are all-zero.

    The element of largest magnitude, with its sign, goes onto the end of the grid
    with the larger magnitude: llama.cpp maps it onto -nmax of [-nmax, nmax - 1],
    which gives a negative scale whenever it is positive. A positive scale maps it
    onto the end of its own sign instead. An unsigned grid only reaches the
    positive weights, so its seed maps the largest of them onto code_max.
    """
    if code_min == 0:
        peak = x.max(dim=-1).values
        bound = float(code_max)
        empty = peak < _ZERO_EPS
    else:
        magnitude, argmax = x.abs().max(dim=-1)
        peak = x.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
        empty = magnitude < _ZERO_EPS
        if positive_scale:
            bound = torch.where(
                peak < 0, torch.full_like(peak, code_min), torch.full_like(peak, code_max))
        else:
            bound = float(code_min if -code_min >= code_max else code_max)
    peak = torch.where(empty, torch.ones_like(peak), peak)
    # Trials move the bound away from zero, -(nmax + offset) in llama.cpp.
    sign = torch.sign(bound) if isinstance(bound, torch.Tensor) else (1. if bound > 0 else -1.)
    return lambda offset: (bound + sign * offset) / peak, empty


def _coordinate_descent(
        quad: torch.Tensor,
        lin: torch.Tensor,
        codes: torch.Tensor,
        code_min: int,
        code_max: int,
        passes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move one code at a time to the value the others imply, while the fit improves.

    The fit is the scale through the origin minimising the objective,
    s = sumlx / suml2 with sumlx = sum(lin * codes) and suml2 = sum(quad * codes^2),
    which leaves the objective at -sumlx^2 / suml2. A move is accepted when it
    raises sumlx^2 / suml2. Returns (that scale, the codes).
    """
    target = _safe_divide(lin, quad)  # the weights themselves, for the plain objective
    sumlx = (lin * codes).sum(-1)
    suml2 = (quad * codes * codes).sum(-1)
    for _ in range(passes):
        changed = False
        for i in range(codes.shape[-1]):
            quad_i, lin_i, code_i = quad[..., i], lin[..., i], codes[..., i]
            without_lx = sumlx - lin_i * code_i
            without_l2 = suml2 - quad_i * code_i * code_i
            usable = (without_lx > 0) & (without_l2 > 0)
            candidate = torch.clamp(
                torch.round(target[..., i] * _safe_divide(without_l2, without_lx)),
                code_min,
                code_max)
            with_lx = without_lx + lin_i * candidate
            with_l2 = without_l2 + quad_i * candidate * candidate
            accept = usable & (candidate
                               != code_i) & (with_lx * with_lx * suml2 > sumlx * sumlx * with_l2)
            if not bool(accept.any()):
                continue
            changed = True
            codes[..., i] = torch.where(accept, candidate, code_i)
            sumlx = torch.where(accept, with_lx, sumlx)
            suml2 = torch.where(accept, with_l2, suml2)
        if not changed:
            break
    return _safe_divide(sumlx, suml2), codes


def symmetric_seed(
        x: torch.Tensor,
        code_min: int,
        code_max: int,
        positive_scale: bool = False) -> torch.Tensor:
    """The absmax scale alone, which is what llama.cpp's reference paths store."""
    inverse_at, empty = _seed(x, code_min, code_max, positive_scale)
    scale = 1.0 / inverse_at(0.)
    return torch.where(empty, torch.zeros_like(scale), scale)


def symmetric_scale_search(
        x: torch.Tensor,
        quad: torch.Tensor,
        lin: torch.Tensor,
        code_min: int,
        code_max: int,
        n_trials: int = 9,
        trial_step: float = 0.1,
        positive_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """llama.cpp `make_qx_quants`: the weighted fit through the origin.

    Codes are rounded at the absmax seed, the scale refit to the minimiser of the
    objective, s = sum(lin * l) / sum(quad * l^2), and then 2 * n_trials trial
    inverse scales are tried around the seed, the bound moved by trial_step * i,
    each refit in turn. The trial that maximises sumlx^2 / suml2, which minimises
    the objective of its refit, wins.

    Returns (scale, codes).
    """
    inverse_at, empty = _seed(x, code_min, code_max, positive_scale)

    def fit(inverse):
        codes = torch.clamp(torch.round(inverse.unsqueeze(-1) * x), code_min, code_max)
        return codes, (lin * codes).sum(-1), (quad * codes * codes).sum(-1)

    codes, sumlx, suml2 = fit(inverse_at(0.))
    scale = _safe_divide(sumlx, suml2)
    best = scale * sumlx
    for trial in range(-n_trials, n_trials + 1):
        if trial == 0:
            continue
        trial_codes, trial_sumlx, trial_suml2 = fit(inverse_at(trial_step * trial))
        accept = (trial_suml2 > 0) & (trial_sumlx * trial_sumlx > best * trial_suml2)
        trial_scale = _safe_divide(trial_sumlx, trial_suml2)
        codes = torch.where(accept.unsqueeze(-1), trial_codes, codes)
        scale = torch.where(accept, trial_scale, scale)
        best = torch.where(accept, trial_scale * trial_sumlx, best)
    scale = torch.where(empty, torch.zeros_like(scale), scale)
    codes = torch.where(empty.unsqueeze(-1), torch.zeros_like(codes), codes)
    return scale, codes


def symmetric_descent_search(
        x: torch.Tensor,
        quad: torch.Tensor,
        lin: torch.Tensor,
        code_min: int,
        code_max: int,
        passes: int = 5,
        positive_scale: bool = False) -> torch.Tensor:
    """llama.cpp `make_q3_quants`: refit at the absmax seed, then coordinate descent.

    llama.cpp weighs every element by its own square here, i.e. quad = x^2.
    """
    inverse_at, empty = _seed(x, code_min, code_max, positive_scale)
    codes = torch.clamp(torch.round(inverse_at(0.).unsqueeze(-1) * x), code_min, code_max)
    scale, _ = _coordinate_descent(quad, lin, codes, code_min, code_max, passes)
    return torch.where(empty, torch.zeros_like(scale), scale)


def affine_seed(x: torch.Tensor, nmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The block range alone, which is what llama.cpp's reference paths store.

    The offset is kept non-positive, as the search keeps it: a quantizer stores
    the magnitude of its zero-point.

    Returns (scale, offset), for codes in [0, nmax].
    """
    block_min = torch.clamp(x.min(dim=-1).values, max=0.)
    return (x.max(dim=-1).values - block_min) / nmax, block_min


def affine_scale_search(
        x: torch.Tensor,
        quad: torch.Tensor,
        lin: torch.Tensor,
        nmax: int,
        n_steps: int = 36,
        trial_step: float = 0.05,
        origin: float = -0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """llama.cpp `make_qkx3_quants`: the weighted affine fit, codes in [0, nmax].

    For fixed codes, the scale and the offset solve the 2x2 system that minimises
    the objective. Codes are rounded at n_steps + 1 trial inverse scales
    `(origin + trial_step * i + nmax) / (max - offset)`, each pair refit, and the
    pair with the lowest objective wins. The offset is kept non-positive
    throughout: a trial whose refit offset is positive falls back to the fit
    through the origin.

    Returns (scale, offset).
    """
    block_min = torch.clamp(x.min(dim=-1).values, max=0.)
    block_max = x.max(dim=-1).values
    empty = block_max <= block_min
    sum_q = quad.sum(-1)
    sum_lin = lin.sum(-1)

    def span(offset):
        # Recomputed from the current offset, so an accepted trial moves the
        # trial scales of the ones after it.
        return torch.where(empty, torch.ones_like(offset), block_max - offset)

    def objective(scale, offset, codes):
        r = scale.unsqueeze(-1) * codes + offset.unsqueeze(-1)
        return (quad * r * r - 2 * lin * r).sum(-1)

    offset = block_min
    scale = span(offset) / nmax
    codes = torch.clamp(torch.round((x - offset.unsqueeze(-1)) / scale.unsqueeze(-1)), 0, nmax)
    best = objective(scale, offset, codes)
    for trial in range(n_steps + 1):
        inverse = (origin + trial_step * trial + nmax) / span(offset)
        v = inverse.unsqueeze(-1) * (x - offset.unsqueeze(-1))
        trial_codes = torch.where(v > 0, torch.round(torch.clamp(v, max=nmax)), torch.zeros_like(v))
        sum_l = (quad * trial_codes).sum(-1)
        sum_l2 = (quad * trial_codes * trial_codes).sum(-1)
        sum_xl = (lin * trial_codes).sum(-1)
        det = sum_q * sum_l2 - sum_l * sum_l
        trial_scale = _safe_divide(sum_q * sum_xl - sum_lin * sum_l, det)
        trial_offset = _safe_divide(sum_l2 * sum_lin - sum_l * sum_xl, det)
        through_origin = trial_offset > 0
        trial_scale = torch.where(through_origin, _safe_divide(sum_xl, sum_l2), trial_scale)
        trial_offset = torch.where(through_origin, torch.zeros_like(trial_offset), trial_offset)
        trial_objective = objective(trial_scale, trial_offset, trial_codes)
        accept = (det > 0) & (trial_objective < best)
        scale = torch.where(accept, trial_scale, scale)
        offset = torch.where(accept, trial_offset, offset)
        best = torch.where(accept, trial_objective, best)
    scale = torch.where(empty, torch.zeros_like(scale), scale)
    offset = torch.where(empty, block_min, offset)
    return scale, offset


def positive_grid_search(
        x: torch.Tensor,
        w: torch.Tensor,
        nmax: int,
        n_trials: int = 4,
        trial_step: float = 0.1,
        passes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """llama.cpp `make_qp_quants`: place non-negative values on a grid `step * [0, nmax]`,
    weighing the squared error of each value by w.

    The inverse step is seeded at `nmax / max` and 2 * n_trials trials
    `(nmax + trial_step * i) / max` are scored by their weighted squared error.
    The codes are then rounded at the winner, the step refit by weighted least
    squares, and polished by coordinate descent.

    Returns (step, codes).
    """
    peak = torch.clamp(x.max(dim=-1).values, min=0.)
    empty = peak < _ZERO_EPS
    peak = torch.where(empty, torch.ones_like(peak), peak)

    def error(inverse):
        codes = torch.clamp(torch.round(inverse.unsqueeze(-1) * x), max=nmax)
        residual = x - codes / inverse.unsqueeze(-1)
        return (w * residual * residual).sum(-1)

    inverse = nmax / peak
    best = error(inverse)
    for trial in range(-n_trials, n_trials + 1):
        if trial == 0:
            continue
        trial_inverse = (nmax + trial_step * trial) / peak
        trial_error = error(trial_inverse)
        accept = trial_error < best
        inverse = torch.where(accept, trial_inverse, inverse)
        best = torch.where(accept, trial_error, best)
    codes = torch.clamp(torch.round(inverse.unsqueeze(-1) * x), max=nmax)
    step, codes = _coordinate_descent(w, w * x, codes, 0, nmax, passes)
    step = torch.where(empty, torch.zeros_like(step), step)
    codes = torch.where(empty.unsqueeze(-1), torch.zeros_like(codes), codes)
    return step, codes


def signed_absmax_grid(values: torch.Tensor, code_min: int,
                       code_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place signed values on a grid `step * [code_min, code_max]` by their signed peak,
    as the symmetric seed does. Returns (step, codes)."""
    inverse_at, empty = _seed(values, code_min, code_max, positive_scale=False)
    inverse = inverse_at(0.)
    codes = torch.clamp(torch.round(inverse.unsqueeze(-1) * values), code_min, code_max)
    step = torch.where(empty, torch.zeros_like(inverse), 1. / inverse)
    codes = torch.where(empty.unsqueeze(-1), torch.zeros_like(codes), codes)
    return step, codes


def positive_absmax_grid(values: torch.Tensor, nmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place non-negative values on a grid `step * [0, nmax]` by their maximum.

    Returns (step, codes).
    """
    peak = values.max(dim=-1).values
    codes = torch.clamp(torch.round(_safe_divide(nmax * values, peak.unsqueeze(-1))), 0, nmax)
    return peak / nmax, codes
