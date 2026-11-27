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

"""Block-wise scale searches, adapted from llama.cpp, as grid optimization solvers.

Each solver runs one of the block searches of .kernels on whatever layer grid
optimization hands it, and every layer the same way. What a layer contributes is
read from its quantizer:

  - the codes span the quantizer's integer grid [code_min, code_max];
  - the fit has an offset when the quantizer has a zero-point, which the offset
    is written to, and goes through the origin otherwise;
  - a block is one of the quantizer's groups, or the whole row of a per-channel
    quantizer;
  - a quantizer that quantizes its scales (and zero-points) in turn on a nested
    grid, a step shared by a run of groups times an integer per group, as the
    GGUF K formats do, has them placed on that grid (see SCALE_PLACEMENTS);
  - a scale that cannot be negative starts from a positive absmax scale, instead
    of the signed one.

A solver only accepts the layers its fit can be written into (see check_layer);
the others are skipped, as grid optimization skips any unsupported layer.

The fit minimises the objective grid optimization hands every solver,
r^T H r - 2 r^T b with the diagonal of H, each element further weighed by a
magnitude term (see objective.py). When X_tilde = X (G = H), that is the
weighted squared error sum(diag(H) * magnitude * (w - r)^2), up to a constant;
diag(H) is twice the mean square activation of each input channel (an
importance matrix), a factor that does not change the fit.
"""

from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers import ScaleSolution
from brevitas.graph.grid_optimization.solvers import ScaleSolverFamily
from brevitas.graph.grid_optimization.utils import _build_grid
from brevitas.graph.grid_optimization.utils import _get_scale_quant_grid
from brevitas.graph.grid_optimization.utils import _get_zero_point_quant_grid
from brevitas.graph.grid_optimization.utils import _has_base_zero_point
from brevitas.graph.grid_optimization.utils import _has_zero_zero_point
from brevitas.graph.grid_optimization.utils import _is_int_weight_quant
from brevitas.graph.grid_optimization.utils import NestedGrid
from brevitas_examples.llm.gguf_export.search.kernels import absmax_scale
from brevitas_examples.llm.gguf_export.search.kernels import affine_scale_search
from brevitas_examples.llm.gguf_export.search.kernels import minmax_scale
from brevitas_examples.llm.gguf_export.search.kernels import positive_absmax_grid
from brevitas_examples.llm.gguf_export.search.kernels import positive_grid_search
from brevitas_examples.llm.gguf_export.search.kernels import signed_absmax_grid
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_descent_search
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_scale_search
from brevitas_examples.llm.gguf_export.search.objective import objective_weights

__all__ = ["SCALE_PLACEMENTS"]

# How the scales of a layer with a nested scale grid are placed on it: by their
# largest magnitude alone, or by a weighted sweep around it.
SCALE_PLACEMENTS = ('absmax', 'sweep')


def _place(values: torch.Tensor, weights: torch.Tensor, grid: NestedGrid,
           placement: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place per-group values [..., groups_per_step] on a nested grid, weighing each
    group by weights.

    Returns (step, codes): one step per run of groups, one integer per group.
    """
    if grid.min_int < 0:
        if placement == 'absmax':
            return signed_absmax_grid(values, grid.min_int, grid.max_int)
        return symmetric_scale_search(values, weights, weights * values, grid.min_int, grid.max_int)
    if placement == 'absmax':
        return positive_absmax_grid(values, grid.max_int)
    return positive_grid_search(values, weights, grid.max_int)


class _GgufSolverFamily(ScaleSolverFamily):
    """A block search, run on whatever layer it is given.

    It uses the diagonal of the covariance matrix only, and fits one scale (and
    zero-point) per block. Subclasses say which search they run.
    """

    # Whether the search can fit a zero-point, for a quantizer with one.
    fits_zero_point: bool = True

    @classmethod
    def _fit(
            cls,
            x: torch.Tensor,
            quad: torch.Tensor,
            lin: torch.Tensor,
            code_min: int,
            code_max: int,
            fit_zero_point: bool,
            positive_scale: bool,
            sweep_steps: Optional[int],
            sweep_step: Optional[float]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fit every block of x. Returns (scale, offset or None), the offset being
        the zero-point in weight units (see ScaleSolution)."""
        raise NotImplementedError

    @classmethod
    def check_layer(cls, layer: nn.Module) -> Optional[str]:
        weight_quant = layer.weight_quant
        if not _is_int_weight_quant(weight_quant):
            return "the GGUF solvers need an integer weight quantizer"
        fit_zero_point = not _has_zero_zero_point(layer)
        scale_quant_grid = _get_scale_quant_grid(layer)
        zero_point_quant_grid = _get_zero_point_quant_grid(layer)
        if fit_zero_point:
            if not cls.fits_zero_point:
                return "this search fits no zero-point, so it needs a quantizer with a zero one"
            if zero_point_quant_grid is None and not _has_base_zero_point(layer):
                return (
                    "a zero-point fit needs a stored zero-point that is not rounded to an "
                    "integer (a ParameterFromStatsFromParameterZeroPoint with "
                    "quantize_zero_point=False)")
            if (scale_quant_grid is None) != (zero_point_quant_grid is None):
                return (
                    "a zero-point fit places the scales and the zero-points on nested grids "
                    "together, so it needs both quantized on one, or neither")
        if scale_quant_grid is None:
            return None
        if zero_point_quant_grid is not None and (zero_point_quant_grid.groups_per_step
                                                  != scale_quant_grid.groups_per_step):
            return "the scales and the zero-points need to share their steps in the same runs"
        grid = _build_grid(layer, dtype=torch.float32, device='cpu')
        signed_scales = (
            not fit_zero_point and bool(grid[0] < 0) and
            not weight_quant.restrict_scale_positive and bool(grid[-1] != -grid[0]))
        if signed_scales and scale_quant_grid.min_int >= 0:
            return "the fit produces signed scales, which the layer's scale grid cannot hold"
        return None

    @classmethod
    def solve(
            cls,
            w: torch.Tensor,
            b: torch.Tensor,
            H: torch.Tensor,
            unscaled_grid: torch.Tensor,
            *,
            allow_negative_s: bool,
            fit_zero_point: bool = False,
            scale_quant_grid: Optional[NestedGrid] = None,
            zero_point_quant_grid: Optional[NestedGrid] = None,
            magnitude: str = 'rms_quad',
            magnitude_factor: float = 1.0,
            magnitude_scope: str = 'row',
            sweep_steps: Optional[int] = None,
            sweep_step: Optional[float] = None,
            scale_placement: str = 'sweep',
            **kwargs) -> ScaleSolution:
        if scale_placement not in SCALE_PLACEMENTS:
            raise ValueError(
                f"unknown scale placement {scale_placement!r}; expected one of "
                f"{list(SCALE_PLACEMENTS)}")
        rows, groups, block = w.shape
        code_min, code_max = int(unscaled_grid[0]), int(unscaled_grid[-1])
        x, h = w, H.reshape(groups, block)
        if scale_quant_grid is not None:
            shape = (rows, -1, scale_quant_grid.groups_per_step, block)
            x, b, h = w.reshape(shape), b.reshape(shape), h.reshape(shape[1:])
        quad, lin = objective_weights(x, b, h, magnitude, magnitude_factor, magnitude_scope)
        scale, offset = cls._fit(
            x, quad, lin, code_min, code_max, fit_zero_point, not allow_negative_s,
            sweep_steps, sweep_step)
        if scale_quant_grid is None:
            return ScaleSolution(scale=scale, offset=offset)

        # The placement of a group's scale is weighed by the total weight of its fit.
        group_weights = quad.sum(-1)
        scale_step, scale_codes = _place(scale, group_weights, scale_quant_grid, scale_placement)
        scale = scale_step.unsqueeze(-1) * scale_codes
        offset_step = None
        if offset is not None:
            # The offsets are non-positive and placed by their magnitude.
            offset_step, offset_codes = _place(
                -offset, group_weights, zero_point_quant_grid, scale_placement)
            offset = -offset_step.unsqueeze(-1) * offset_codes
        return ScaleSolution(
            scale=scale, offset=offset, scale_step=scale_step, offset_step=offset_step)


@SCALE_SOLVER_REGISTRY.register(names='gguf-minmax')
class GgufMinMaxSolverFamily(_GgufSolverFamily):
    """No search: the absmax scale of a block, or its min-max scale and zero-point
    for a quantizer with a zero-point."""

    @classmethod
    def _fit(cls, x, quad, lin, code_min, code_max, fit_zero_point, positive_scale, *args):
        if fit_zero_point:
            return minmax_scale(x, code_max - code_min)
        return absmax_scale(x, code_min, code_max, positive_scale), None


@SCALE_SOLVER_REGISTRY.register(names='gguf-sweep')
class GgufSweepSolverFamily(_GgufSolverFamily):
    """Trial-scale sweep. Without a zero-point, a weighted fit through the origin,
    refit at the absmax scale and at trial scales on both sides of it; with one, a
    weighted fit of a scale and a zero-point at trial scales across the block's
    range. The fit with the lowest objective wins."""

    @classmethod
    def _fit(
            cls,
            x,
            quad,
            lin,
            code_min,
            code_max,
            fit_zero_point,
            positive_scale,
            sweep_steps,
            sweep_step):
        if fit_zero_point:
            return affine_scale_search(
                x,
                quad,
                lin,
                code_max - code_min,
                n_steps=36 if sweep_steps is None else sweep_steps,
                trial_step=0.05 if sweep_step is None else sweep_step)
        scale, _ = symmetric_scale_search(
            x,
            quad,
            lin,
            code_min,
            code_max,
            n_trials=9 if sweep_steps is None else sweep_steps,
            trial_step=0.1 if sweep_step is None else sweep_step,
            positive_scale=positive_scale)
        return scale, None


@SCALE_SOLVER_REGISTRY.register(names='gguf-descent')
class GgufDescentSolverFamily(_GgufSolverFamily):
    """Weighted fit through the origin, refined by coordinate descent on the codes:
    refit at the absmax scale, then move one code at a time while the fit improves.
    It has no trial scales and fits no zero-point."""
    fits_zero_point = False

    @classmethod
    def _fit(cls, x, quad, lin, code_min, code_max, fit_zero_point, positive_scale, *args):
        return symmetric_descent_search(
            x, quad, lin, code_min, code_max, positive_scale=positive_scale), None
