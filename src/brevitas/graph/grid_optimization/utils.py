# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Grid builders, scale accessors and ordering helpers for grid optimization.

Weight-independent utilities: the integer/minifloat quantization grid builders,
the grid/threshold derivation from a layer's quantizer (via its proxy
accessors), the read/write accessors for the stored scale and zero-point
parameters (including those quantized in turn on a nested grid), and the
group-aware importance ordering used by the sequential/interleaved paths.

All of these take the quant layer (or its weight quantizer) and unpack it here,
so the reach through the proxy into tensor_quant lives in exactly one place
(_has_base_scale) rather than being spread over the algorithm code.
"""

from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Optional
import warnings

import torch
import torch.nn as nn

from brevitas.core.function_wrapper import CeilSte
from brevitas.core.function_wrapper import FloorSte
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
from brevitas.core.zero_point import _ScaleShiftZeroPoint
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
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
    """Whether layer stores a writable scale parameter grid optimization can optimize.

    True only for an enabled weight quantizer whose scaling_impl is a
    ParameterFromStatsFromParameterScaling, i.e. the one scaling implementation
    that keeps the scale as a plain stored Parameter. This is the single place
    that reaches through the proxy into tensor_quant; the accessors below and
    GridOptimizer.is_layer_valid all go through it.
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
def _set_base_scale(layer: nn.Module, value: torch.Tensor, groups: slice = slice(None)) -> None:
    """Write value into layer's stored scale Parameter, for the given groups.

    The write is in place so the registered Parameter keeps its identity. A
    per-channel scale is a single group.
    """
    if not _has_base_scale(layer):
        raise RuntimeError(
            "Cannot set base scale: weight scale is not a "
            "ParameterFromStatsFromParameterScaling.")
    target = layer.weight_quant.tensor_quant.scaling_impl.value[:, groups]
    target[...] = value.reshape(target.shape)


def _has_base_zero_point(layer: nn.Module) -> bool:
    """Whether layer stores a writable, real-valued zero-point parameter.

    True for a ParameterFromStatsFromParameterZeroPoint that does not round the
    zero-point to an integer, the one implementation a solver's zero-point can be
    written into exactly.
    """
    zero_point_impl = layer.weight_quant.tensor_quant.zero_point_impl
    if not isinstance(zero_point_impl, ParameterFromStatsFromParameterZeroPoint):
        return False
    shift = zero_point_impl.scale_shift_zero_point
    return isinstance(shift, _ScaleShiftZeroPoint) and not shift.quantize_zero_point


def _stored_zero_point(offset: torch.Tensor) -> torch.Tensor:
    """What a zero-point Parameter holds for a given offset, the reconstruction of the
    lowest code (see ScaleSolution).

    The quantizer reconstructs (q - zero_point) * scale with zero_point =
    stored / scale + q_min, i.e. (q - q_min) * scale - stored, so it holds -offset.
    It takes the magnitude of what it holds, so a positive offset cannot be
    represented.
    """
    if (offset > 0).any():
        raise ValueError(
            "Cannot set zero-point: the quantizer stores the magnitude of its zero-point, "
            "so a positive offset cannot be represented.")
    return -offset


@torch.no_grad()
def _set_base_zero_point(
    layer: nn.Module, offset: torch.Tensor, groups: slice = slice(None)) -> None:
    """Write the zero-point of a given offset (see _stored_zero_point) into layer's stored
    zero-point Parameter, for the given groups."""
    if not _has_base_zero_point(layer):
        raise RuntimeError(
            "Cannot set base zero-point: the weight zero-point is not a "
            "ParameterFromStatsFromParameterZeroPoint with quantize_zero_point disabled.")
    target = layer.weight_quant.tensor_quant.zero_point_impl.value[:, groups]
    target[...] = _stored_zero_point(offset).reshape(target.shape)


@dataclass(frozen=True)
class NestedGrid:
    """The integer grid a layer's per-group scales (or zero-points) are quantized on.

    Some quantizers quantize their scale (or zero-point) in turn on every forward
    pass, with a nested quantizer: every run of groups_per_step consecutive
    groups shares one step, and the value of a group is that step times an
    integer in [min_int, max_int], which the nested quantizer derives from the
    stored value with its rounding ('round', 'ceil' or 'floor').
    """
    groups_per_step: int
    min_int: int
    max_int: int
    rounding: str


_ROUNDINGS = {RoundSte: 'round', CeilSte: 'ceil', FloorSte: 'floor'}

# Where each rounding reads an index back from most robustly: the middle of the
# interval of values it maps onto that index.
_ROUNDING_CENTRES = {'round': 0., 'ceil': -0.5, 'floor': 0.5}


def _get_scale_quant(layer: nn.Module) -> Optional[nn.Module]:
    """The nested quantizer layer's stored scale goes through on every forward, or None.

    Requires _has_base_scale(layer). The scale is quantized in turn when it is
    restricted by a QuantRestrictValue, whose quantizer maps it onto its own grid.
    """
    stats_scaling_impl = layer.weight_quant.tensor_quant.scaling_impl.stats_scaling_impl
    restrict = stats_scaling_impl.restrict_clamp_scaling.restrict_value_impl
    return restrict.float_to_int_impl if isinstance(restrict, QuantRestrictValue) else None


def _get_zero_point_quant(layer: nn.Module) -> Optional[nn.Module]:
    """The nested quantizer layer's stored zero-point goes through on every forward, or None."""
    zero_point_impl = layer.weight_quant.tensor_quant.zero_point_impl
    if not isinstance(zero_point_impl, ParameterFromStatsFromParameterZeroPoint):
        return None
    shift = zero_point_impl.scale_shift_zero_point
    return shift.zp_int_quant if isinstance(shift, _ScaleShiftQuantZeroPoint) else None


def _nested_grid(stored: torch.Tensor, quant: Optional[nn.Module]) -> Optional[NestedGrid]:
    """The grid quant places the stored values on, or None without a nested quantizer.

    Writing into it (see _write_on_nested_grid) takes an integer quantizer with a
    stored step and a rounding whose intervals are known; any other nested
    quantizer is rejected.
    """
    if quant is None:
        return None
    rounding = None
    if isinstance(quant, RescalingIntQuant) and isinstance(quant.scaling_impl,
                                                           ParameterFromStatsFromParameterScaling):
        rounding = _ROUNDINGS.get(type(quant.int_quant.float_to_int_impl))
    if rounding is None:
        raise ValueError(
            "Grid optimization can only write into a nested quantizer that is a "
            "RescalingIntQuant with a stored step and round, ceil or floor rounding.")
    bit_width = quant.msb_clamp_bit_width_impl()
    return NestedGrid(
        groups_per_step=stored.numel() // quant.scaling_impl.value.numel(),
        min_int=int(quant.int_quant.min_int(bit_width)),
        max_int=int(quant.int_quant.max_int(bit_width)),
        rounding=rounding)


def _get_scale_quant_grid(layer: nn.Module) -> Optional[NestedGrid]:
    """The nested grid layer's scales are quantized on, or None.

    Requires _has_base_scale(layer).
    """
    return _nested_grid(_get_base_scale(layer), _get_scale_quant(layer))


def _get_zero_point_quant_grid(layer: nn.Module) -> Optional[NestedGrid]:
    """The nested grid layer's zero-points are quantized on, or None."""
    quant = _get_zero_point_quant(layer)
    if quant is None:
        return None
    return _nested_grid(layer.weight_quant.tensor_quant.zero_point_impl.value, quant)


def _write_on_nested_grid(
        stored: nn.Parameter,
        quant: nn.Module,
        grid: NestedGrid,
        value: torch.Tensor,
        step: torch.Tensor,
        read_back: Callable[[torch.Tensor], torch.Tensor],
        groups: slice) -> int:
    """Write value, a step times an integer per group, so that quant reproduces it.

    stored is the parameter quant re-quantizes on every forward pass, which hands
    it read_back(stored). value has one entry per group in groups, a whole number
    of runs of grid.groups_per_step, and step one per run, both in the units of
    stored. The step goes into quant's own parameter first. stored then gets each
    integer times the step quant actually divides by, moved to the middle of the
    interval quant's rounding maps onto that integer, so that the precision stored
    is held in cannot tip it into a neighbour. Zero is written as zero, which every
    rounding reads back exactly.

    Returns how many groups quant does not read back as written.
    """
    steps = slice(groups.start // grid.groups_per_step, groups.stop // grid.groups_per_step)
    rows = stored.shape[0]
    step = step.to(stored.device).reshape(rows, -1, 1)
    value = value.to(stored.device).reshape(rows, step.shape[1], -1)
    live = step != 0
    index = value / torch.where(live, step, torch.ones_like(step))
    index = torch.where(live, index, torch.zeros_like(index))
    index = index.round().clamp(grid.min_int, grid.max_int)

    bit_width = quant.msb_clamp_bit_width_impl()
    threshold = quant.int_scaling_impl(bit_width)
    stored_step = quant.scaling_impl.value[:, steps]
    stored_step[...] = (step * threshold).reshape(stored_step.shape)
    applied_step = quant.scaling_impl(stored, threshold)[:, steps].reshape(rows, -1, 1)
    centred = torch.where(index == 0, index, index + _ROUNDING_CENTRES[grid.rounding])
    target = stored[:, groups]
    target[...] = (applied_step * centred).reshape(target.shape)

    quantized, read_step, *_ = quant(read_back(stored))
    read_index = torch.round(quantized[:, steps] / read_step[:, steps]).reshape(index.shape)
    return int((read_index != index).sum())


def _warn_misread(misread: int, total: int, what: str) -> None:
    if misread:
        warnings.warn(
            f"{misread} of {total} {what} were read back on a neighbouring point of their "
            "grid: the precision the quantizer stores them in cannot hold their integers "
            "exactly. Those groups are one grid step away from the solver's choice.")


@torch.no_grad()
def _set_nested_scale(
        layer: nn.Module, value: torch.Tensor, step: torch.Tensor, groups: slice) -> None:
    """Write the scales of the given groups, which layer's quantizer quantizes in turn
    (see NestedGrid).

    value is what the stored scale parameter holds (the scale times its
    threshold) and step is the shared step of each run of groups, in the same
    units.
    """
    misread = _write_on_nested_grid(
        layer.weight_quant.tensor_quant.scaling_impl.value,
        _get_scale_quant(layer),
        _get_scale_quant_grid(layer),
        value,
        step,
        read_back=lambda stored: stored,
        groups=groups)
    _warn_misread(misread, value.numel(), 'scales')


@torch.no_grad()
def _set_nested_zero_point(
        layer: nn.Module, offset: torch.Tensor, step: torch.Tensor, groups: slice) -> None:
    """Write the zero-points of the given groups, which layer's quantizer quantizes in
    turn (see NestedGrid).

    offset is the reconstruction of the lowest code (see _stored_zero_point), and
    step the shared step of each run of groups of -offset.
    """
    stored = _stored_zero_point(offset)
    # The quantizer takes the magnitude of the stored value before quantizing it.
    misread = _write_on_nested_grid(
        layer.weight_quant.tensor_quant.zero_point_impl.value,
        _get_zero_point_quant(layer),
        _get_zero_point_quant_grid(layer),
        stored,
        step,
        read_back=torch.abs,
        groups=groups)
    _warn_misread(misread, stored.numel(), 'zero-points')


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
    """Group-aware importance ordering over a covariance diagonal x.

    Splits x into contiguous blocks of group_size and orders whole blocks
    by aggregated score (sum or max of diag(H) per block), keeping each block's
    weights contiguous. Used when grid optimization is interleaved group-wise
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
