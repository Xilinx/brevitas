# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Grid builders, scale accessors and ordering helpers for PiSO.

Weight-independent utilities: the integer/minifloat quantization grid builders,
the grid/threshold derivation from a layer's quantizer (via its proxy
accessors), the read/write accessors for the stored scale parameter, and the
group-aware importance ordering used by the sequential/interleaved paths.

All of these take the quant layer (or its weight quantizer) and unpack it here,
so the reach through the proxy into tensor_quant lives in exactly one place
(_has_base_scale) rather than being spread over the algorithm code.
"""

from typing import Any
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


def make_int_grid(bit_width: int, signed: bool = True, narrow_range: bool = False) -> List[float]:
    """Return the sorted integer quantization grid, e.g. [-8, ..., 7] for int4."""
    max_clamp = (1 << (bit_width - signed)) - 1
    min_clamp = signed * -1 * ((1 << (bit_width - 1)) - narrow_range)
    return [float(val) for val in range(min_clamp, max_clamp + 1, 1)]


def make_fp_grid(
        e_bits: int,
        m_bits: int,
        exponent_bias: Optional[float] = None,
        signed: bool = True) -> List[float]:
    """Return the sorted minifloat grid for the given exponent/mantissa widths."""
    if exponent_bias is None:
        exponent_bias = (1 << (e_bits - 1)) - 1
    M = 1 << m_bits
    E = 1 << e_bits
    S = 1 << signed

    grid = set()
    for e in range(E):
        exponent_value = e + (not e)
        for m in range(M):
            mantissa_fixed = m / M + bool(e)
            for sign in range(S):
                grid.add(
                    ((-1.) ** sign) * (mantissa_fixed) * (2.0 ** (exponent_value - exponent_bias)))

    grid = list(grid)
    grid.sort()
    return grid


def _is_float_weight_quant(weight_quant: nn.Module) -> bool:
    """Whether the weight quantizer is a minifloat one.

    Both the plain and groupwise float weight proxies derive from
    WeightFloatQuantProxyFromInjectorBase, so an isinstance check tells int and
    float weight quantizers apart without touching quant_injector.
    """
    return isinstance(weight_quant, WeightFloatQuantProxyFromInjectorBase)


def _is_int_weight_quant(weight_quant: nn.Module) -> bool:
    """Whether the weight quantizer is an integer one.

    Both the plain and groupwise int weight proxies derive from
    WeightQuantProxyFromInjector, whereas the float proxies derive from
    WeightQuantProxyFromInjectorBase (not the concrete int class), so an
    isinstance check tells them apart without touching quant_injector.
    """
    return isinstance(weight_quant, WeightQuantProxyFromInjector)


def _has_base_scale(layer: nn.Module) -> bool:
    """Whether layer stores a writable scale parameter PiSO can optimize.

    True only for an enabled weight quantizer whose scaling_impl is a
    ParameterFromStatsFromParameterScaling, i.e. the one scaling implementation
    that keeps the scale as a plain stored Parameter. This is the single place
    that reaches through the proxy into tensor_quant; the accessors below and
    ScaleOptimizer.is_layer_valid all go through it.
    """
    weight_quant = getattr(layer, 'weight_quant', None)
    if weight_quant is None or not weight_quant.is_quant_enabled:
        return False
    return isinstance(
        weight_quant.tensor_quant.scaling_impl, ParameterFromStatsFromParameterScaling)


def _has_zero_zero_point(layer: nn.Module) -> bool:
    """Whether layer's weight quantizer has a hardwired zero zero-point.

    PiSO's closed form assumes a symmetric quantizer, so an int quantizer has to
    use ZeroZeroPoint. Kept here next to _has_base_scale so both tensor_quant
    lookups stay in this module.
    """
    return isinstance(layer.weight_quant.tensor_quant.zero_point_impl, ZeroZeroPoint)


def _get_base_scale(layer: nn.Module) -> Optional[torch.Tensor]:
    """Return layer's stored scale Parameter, or None if it has none.

    The Parameter itself is returned (not a copy), so callers can read its shape
    and dtype; write through _set_base_scale rather than mutating it directly.
    """
    if not _has_base_scale(layer):
        return None
    return layer.weight_quant.tensor_quant.scaling_impl.value


@torch.no_grad()
def _set_base_scale(layer: nn.Module, value: torch.Tensor, index: Any = slice(None)) -> None:
    """Write value into layer's stored scale Parameter at index.

    The write is in place so the registered Parameter keeps its identity, and
    index lets a caller update a single group's column (PiSO's group-wise
    updater) instead of the whole tensor.
    """
    if not _has_base_scale(layer):
        raise RuntimeError(
            "Cannot set base scale: weight scale is not a "
            "ParameterFromStatsFromParameterScaling.")
    layer.weight_quant.tensor_quant.scaling_impl.value[index] = value


def _build_grid(layer: nn.Module, dtype: torch.dtype, device) -> torch.Tensor:
    """Build the unscaled quantization grid (intX or float_eXmY) for layer.

    Reads the layer's instantiated weight quantizer (its proxy accessors) to pick
    the grid builder and returns the sorted grid as a tensor on the given
    dtype/device.
    """
    weight_quant = layer.weight_quant
    if _is_float_weight_quant(weight_quant):
        grid = make_fp_grid(
            e_bits=int(weight_quant.exponent_bit_width().item()),
            m_bits=int(weight_quant.mantissa_bit_width().item()),
            exponent_bias=int(weight_quant.exponent_bias().item()),
            signed=weight_quant.is_signed)
    else:
        grid = make_int_grid(
            bit_width=int(weight_quant.bit_width().item()),
            signed=weight_quant.is_signed,
            narrow_range=weight_quant.is_narrow_range)
    return torch.tensor(grid, dtype=dtype, device=device)


def _grid_threshold(grid: torch.Tensor, weight_quant: nn.Module) -> torch.Tensor:
    """Scale threshold that maps an unscaled grid to the quantizer's convention."""
    if _is_float_weight_quant(weight_quant):
        # TODO: adapt this for float, thresholds is max_float here, so maybe it can be gathered from somewhere
        return grid[-1]
    return grid.abs().max()


def compute_group_importance_permutation(
        x: torch.Tensor,
        group_size: int,
        descending: bool = True,
        variant: str = "sum") -> torch.Tensor:
    """Group-aware importance ordering over a Hessian diagonal x.

    Splits x into contiguous blocks of group_size and orders whole blocks
    by aggregated score (sum or max of diag(H) per block), keeping each block's
    weights contiguous. Used when scale optimization is interleaved group-wise
    with error correction, where all weights of a group must be processed before
    the next group. Returns (perm, block_order).
    """
    n = x.numel()
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if n % group_size != 0:
        raise ValueError("length of diagonal must be a multiple of group_size")

    num_blocks = n // group_size
    blocks = x.view(num_blocks, group_size)

    # Sort blocks by sum
    if variant == "sum":
        block_scores = blocks.sum(dim=1)
    elif variant == "max":
        block_scores = blocks.max(dim=1).values
    else:
        raise ValueError("variant must be 'sum' or 'max'")

    order_blocks = torch.argsort(block_scores, descending=descending)

    order_intra_block = torch.argsort(blocks, dim=1, descending=descending)
    perm_parts = [b * group_size + order_intra_block[b] for b in order_blocks]

    perm = torch.cat(perm_parts)
    return perm, order_blocks
