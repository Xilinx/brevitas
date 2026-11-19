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

"""What a fit is weighed by: the covariance of each input channel times a magnitude
term computed from the weights themselves.

llama.cpp's usual magnitude, `rms_quad`, behaves like |x| for large weights,
making the fit relative rather than absolute, and saturates for near-zero
weights so they keep some say.
"""

from typing import Tuple

import torch

__all__ = ["MAGNITUDE_FORMS", "SCOPES", "objective_weights"]

# Each takes the weights and the mean square over the configured scope.
_MAGNITUDES = {
    'one': lambda x,
           mean_sq,
           factor: torch.ones_like(x),
    'square': lambda x,
              mean_sq,
              factor: x * x,
    'abs': lambda x,
           mean_sq,
           factor: x.abs(),
    'sqrt_abs': lambda x,
                mean_sq,
                factor: torch.sqrt(x.abs()),
    'rms_quad': lambda x,
                mean_sq,
                factor: torch.sqrt(factor * mean_sq + x * x),
    'rms_plus_abs': lambda x,
                    mean_sq,
                    factor: torch.sqrt(mean_sq) + x.abs(),}
MAGNITUDE_FORMS = tuple(_MAGNITUDES)

# What the mean square of a magnitude term is taken over: a block (group), the run
# of groups sharing a scale step on a nested grid, or the whole row.
SCOPES = ('block', 'shared_step', 'row')


def objective_weights(
        x: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        magnitude: str = 'rms_quad',
        factor: float = 1.0,
        scope: str = 'row') -> Tuple[torch.Tensor, torch.Tensor]:
    """The weights of the quadratic and of the linear term of the fit.

    The fit minimises sum(quad * r^2) - 2 * sum(lin * r) over the reconstruction r,
    i.e. r^T H r - 2 r^T b with a diagonal H, each element weighed by the magnitude
    term. x is [rows, blocks, block], or [rows, runs, blocks, block] for a layer
    whose blocks share a scale step in runs; b has the same shape, and h, the
    diagonal of the covariance matrix, the shape of x without its rows.
    """
    if scope == 'shared_step' and x.dim() < 4:
        raise ValueError(
            "a 'shared_step' scope needs a quantizer whose scales are quantized on a grid "
            "of their own")
    dims = {'block': (-1,), 'shared_step': (-2, -1), 'row': tuple(range(1, x.dim()))}[scope]
    mean_sq = (x * x).mean(dim=dims, keepdim=True)
    term = _MAGNITUDES[magnitude](x, mean_sq, factor)
    return h * term, b * term
