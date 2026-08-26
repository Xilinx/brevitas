"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Shared quantizer-builder infrastructure: the quantization-axis enums, the
brevitas injector mixins (symmetric / asymmetric zero-point, groupwise
power-of-two, ...) and the MSE / HQO local-loss injector factories, plus the
float-format parsing helpers. These are consumed by :mod:`.core` and
:mod:`.components` and have no dependency on the builder itself, so they sit at
the base of the builder package.
"""
from enum import auto
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import warnings

from dependencies import this
from dependencies import value
from torch import nn

from brevitas.core.function_wrapper.ops_ste import FloorSte
from brevitas.core.function_wrapper.shape import StatsInputViewShapeImpl
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.stats import MSE
from brevitas.core.stats.stats_op import HalfQuadraticOptimizerScale
from brevitas.core.stats.stats_op import HalfQuadraticOptimizerZeroPoint
from brevitas.core.stats.stats_op import MSEUniformStepBase
from brevitas.core.zero_point import ParameterFromRuntimeZeroPoint
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.core.zero_point import ParameterZeroPoint
from brevitas.core.zero_point import StatsFromParameterZeroPoint
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.quant.float_quant_fnuz import FpFNUZMixin
from brevitas.quant.float_quant_ocp import FpOCPMixin
from brevitas.quant.solver.common import solve_stats_impl
from brevitas.utils.float_quant_utils import get_midmax_mantissa_bit_bias
from brevitas.utils.python_utils import AutoName

EnumTypeVar = TypeVar('EnumTypeVar')
EnumType = Optional[Union[str, EnumTypeVar]]

# MSE sub-injectors and mixins are parametrized over the quantity being
# searched ("scale" or "zero_point"). Brevitas uses two separate sub-injectors
# (MSE*ScaleSubInjector / MSEZeroPointSubInjector) to avoid a name clash between
# the scale and the zero-point stats: both rely on a nested injector whose
# `stats_impl` is MSE, but they are wired into different parent attributes
# (`scaling_stats_impl` vs `zero_point_stats_impl`) and through different
# `*_stats_input_view_shape_impl` names.
#
# IMPORTANT (mse_init_op clash): the scale and the zero-point require *different*
# `mse_init_op`. For the scale it is derived from `scaling_stats_op` /
# `restrict_scaling_type` (e.g. AbsMax / AbsMinMax); for the zero-point it is a
# fixed NegativeMinOrZero. A single sub-injector that pulls `mse_init_op` from
# its parent (`(this << 1).mse_init_op`) would therefore wire the scale's init
# op into the zero-point search. To keep the infrastructure shareable we make
# `mse_init_op` an explicit per-target parameter of the sub-injector factory
# instead of inheriting it from the parent.


class MSESubInjectorMixin(ExtendedInjector):
    scaling_per_output = (this << 1).scaling_per_output
    proxy_module = (this << 1).proxy_module
    stats_impl = MSE
    mse_iters = 20
    mse_base_op = MSEUniformStepBase
    stats_reduce_dim = (this << 1).stats_reduce_dim
    device = (this << 1).device
    dtype = (this << 1).dtype
    permute_dims = (this << 1).permute_dims
    inner_stats_input_view_shape_impl = (this << 1).inner_stats_input_view_shape_impl
    mse_search_method = 'grid'

    @value
    def restrict_scale_positive():
        return (this << 1).restrict_scale_positive


class SymMixin(ExtendedInjector):
    zero_point_impl = ZeroZeroPoint
    scaling_stats_op = StatsOp.MAX


class ZeroPointImplType(AutoName):
    ZERO = auto()  # ZeroZeroPoint (symmetric / no zero point)
    STATS = auto()  # StatsFromParameterZeroPoint
    PARAMETER = auto()  # ParameterZeroPoint
    PARAMETER_FROM_STATS = auto()  # ParameterFromStatsFromParameterZeroPoint
    PARAMETER_FROM_RUNTIME = auto()  # ParameterFromRuntimeZeroPoint  (optional, activations)


class AsymmetricZeroPointMixin(ExtendedInjector):
    scaling_stats_op = StatsOp.MIN_MAX
    zero_point_stats_op = StatsOp.NEG_MIN_OR_ZERO
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
    zero_point_stats_input_concat_dim = this.scaling_stats_input_concat_dim

    @value
    def zero_point_stats_impl(zero_point_stats_op=None):
        return solve_stats_impl(zero_point_stats_op)

    # The zero-point implementation is driven by the *zero-point* param method
    # (zero_point_impl_type) when one is selected (MSE / HQO). Otherwise
    # (zero_point_impl_type is None) the default asymmetric zero-point mirrors the
    # *scale* storage strategy (scaling_impl_type), so e.g. a parameter-from-stats
    # scale folds the zero-point into a standalone parameter as well.
    @value
    def zero_point_impl(
            zero_point_impl_type: EnumType[ZeroPointImplType] = None,
            scaling_impl_type: EnumType[ScalingImplType] = None) -> Optional[Type[nn.Module]]:
        if zero_point_impl_type is None:
            if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
                zero_point_impl_type = ZeroPointImplType.PARAMETER_FROM_STATS
            elif scaling_impl_type == ScalingImplType.PARAMETER:
                zero_point_impl_type = ZeroPointImplType.PARAMETER
            else:
                # STATS / AFFINE_STATS / DYNAMIC / ... -> plain stats-from-parameter zp.
                zero_point_impl_type = ZeroPointImplType.STATS
        if zero_point_impl_type == ZeroPointImplType.STATS:
            return StatsFromParameterZeroPoint
        elif zero_point_impl_type == ZeroPointImplType.PARAMETER:
            return ParameterZeroPoint
        elif zero_point_impl_type == ZeroPointImplType.PARAMETER_FROM_STATS:
            return ParameterFromStatsFromParameterZeroPoint
        elif zero_point_impl_type == ZeroPointImplType.PARAMETER_FROM_RUNTIME:
            return ParameterFromRuntimeZeroPoint
        elif zero_point_impl_type == ZeroPointImplType.ZERO:
            return ZeroZeroPoint
        else:
            warnings.warn(
                f"Defaulting to ZeroZeroPoint for unrecognized zero_point_impl_type {zero_point_impl_type}."
            )
            return ZeroZeroPoint


class RestrictThresholdMixin(ExtendedInjector):
    restrict_value_float_to_int_impl = FloorSte
    restrict_scaling_impl = PowerOfTwoRestrictValue


class GroupwisePoTMixin(ExtendedInjector):
    threshold_mixin = RestrictThresholdMixin

    @value
    def restrict_threshold_impl():
        return this.threshold_mixin.restrict_scaling_impl

    @value
    def midmax_mantissa_bit_bias(mantissa_bit_width, nan_values, inf_values):
        return get_midmax_mantissa_bit_bias(mantissa_bit_width, nan_values, inf_values)


@value
def inner_stats_input_view_shape_impl(scaling_per_output):
    if scaling_per_output == ScalingPerOutputType.CHANNEL:
        return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
    elif scaling_per_output == ScalingPerOutputType.TENSOR:
        return StatsInputViewShapeImpl.OVER_TENSOR
    elif scaling_per_output == ScalingPerOutputType.GROUP:
        return StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK


class Target(AutoName):
    SCALE = auto()
    ZERO_POINT = auto()


class ParamMethod(AutoName):
    STATS = auto()
    MSE = auto()
    HQO = auto()


# TODO: Come up with a better name for this enum
class QuantParamType(AutoName):
    SYM = auto()
    ASYM = auto()


class FloatFormat(AutoName):
    # Maps to WEIGHT_QUANT_MAP keys 'float' / 'float_ocp' / 'float_fnuz'.
    FLOAT = auto()  # plain FP (brevitas ScaledFloatWeightBase)
    OCP = auto()  # FpOCPMixin (inf/nan values, max_available_float)
    FNUZ = auto()  # FpFNUZMixin (exponent_bias = 2 ** (e - 1))


# Mapping from FloatFormat to the brevitas format mixin to compose with the
# float weight base. FLOAT requires no extra mixin.
FLOAT_FORMAT_MIXIN_MAP = {
    FloatFormat.OCP.value: FpOCPMixin,
    FloatFormat.FNUZ.value: FpFNUZMixin,}

_FLOAT_QUANT_FORMAT_RE = re.compile(r'^e([1-8])m([1-8])$')


def parse_float_quant_format(float_quant_format: str) -> Tuple[int, int, int]:
    """Parse a float format string such as ``"e4m3"`` into bit widths.

    Returns ``(exponent_bit_width, mantissa_bit_width, bit_width)`` where
    ``bit_width = 1 (sign) + exponent_bit_width + mantissa_bit_width``.
    """
    match = _FLOAT_QUANT_FORMAT_RE.match(float_quant_format)
    if match is None:
        raise ValueError(
            f"Unrecognized float_quant_format {float_quant_format!r}; expected e.g. 'e4m3'.")
    exponent_bit_width = int(match.group(1))
    mantissa_bit_width = int(match.group(2))
    # +1 for the sign bit (float weight quantizers are signed).
    bit_width = 1 + exponent_bit_width + mantissa_bit_width
    return exponent_bit_width, mantissa_bit_width, bit_width


@value
def scale_init_op(
    scaling_stats_op: EnumType[StatsOp] = None,
    restrict_scaling_type: EnumType[RestrictValueType] = None,
) -> Optional[nn.Module]:
    return solve_stats_impl(scaling_stats_op, restrict_scaling_type)


@value
def zero_point_init_op(zero_point_stats_op: EnumType[StatsOp] = None,) -> Optional[nn.Module]:
    return solve_stats_impl(zero_point_stats_op)


def _make_mse_injector(target: Target = Target.SCALE,
                       overrides: Optional[Dict[str, Any]] = None) -> Type[ExtendedInjector]:
    prefix = {
        Target.SCALE.value: "scaling",
        Target.ZERO_POINT.value: "zero_point",}[target.value]
    MSESubInjector = type(
        f'MSE{target.capitalize()}SubInjector', (MSESubInjectorMixin,), {
            "mse_init_op": getattr(this << 1, f"{prefix}_mse_init_op"),})

    def _make_scaling_init_op(prefix: str) -> Callable:
        # The scale init op derives from scaling_stats_op (AbsMax / AbsMinMax /
        # ...), while the zero-point init op derives from zero_point_stats_op
        # (NegativeMinOrZero for asym quantizers). See the mse_init_op clash note.
        init_op = scale_init_op if target == Target.SCALE else zero_point_init_op
        init_op.__name__ = f"{prefix}_mse_init_op"
        return init_op

    namespace = {
        f"mse_{target}": MSESubInjector,
        f"{prefix}_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
        f"{prefix}_stats_input_view_shape_impl": nn.Identity(),
        f"{prefix}_stats_impl": getattr(this, f"mse_{target}").stats_impl,
        f"{prefix}_mse_init_op": _make_scaling_init_op(prefix),
        "inner_stats_input_view_shape_impl": inner_stats_input_view_shape_impl,}

    # Caller can override any of the default namespace entries
    if overrides is not None:
        namespace.update(overrides)

    mse_injector = type(f'MSE{target.capitalize()}Injector', (ExtendedInjector,), namespace)
    return mse_injector


def _make_hqo_injector(target: Target = Target.SCALE) -> Type[ExtendedInjector]:
    if target == Target.SCALE:
        HQOClass = HalfQuadraticOptimizerScale
    elif target == Target.ZERO_POINT:
        HQOClass = HalfQuadraticOptimizerZeroPoint

    prefix = {
        Target.SCALE.value: "scaling",
        Target.ZERO_POINT.value: "zero_point",}[target.value]
    suffix = {
        Target.SCALE.value: "scale",
        Target.ZERO_POINT.value: "zp",}[target.value]

    def _make_init_op() -> Callable:
        # The scale init op derives from scaling_stats_op (AbsMax / AbsMinMax /
        # ...), while the zero-point init op derives from zero_point_stats_op
        # (NegativeMinOrZero for asym quantizers). See the mse_init_op clash note.
        init_op = scale_init_op if target == Target.SCALE else zero_point_init_op
        init_op.__name__ = f"hqo_init_op_{suffix}"
        return init_op

    namespace = {
        f"hqo_{target}": HQOClass,
        f"{prefix}_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
        f"{prefix}_stats_impl": getattr(this, f"hqo_{target}"),
        f"hqo_init_op_{suffix}": _make_init_op(),
        "inner_stats_input_view_shape_impl": getattr(this, f"{prefix}_stats_input_view_shape_impl"),
    }
    hqo_injector = type(f'HQO{target.capitalize()}Injector', (ExtendedInjector,), namespace)
    return hqo_injector


# The init op is derived at the parent injector level from the requested enums
# (scaling_stats_op / zero_point_stats_op) and pulled into the sub-injector via
# `(this << 1)`, with a distinct name per target to avoid the mse_init_op clash.
MSEScaleInjectorMixin = _make_mse_injector(target=Target.SCALE, overrides={'keepdim': False})
MSEZeroPointInjectorMixin = _make_mse_injector(target=Target.ZERO_POINT)

HQOScaleInjectorMixin = _make_hqo_injector(target=Target.SCALE)
HQOZeroPointInjectorMixin = _make_hqo_injector(target=Target.ZERO_POINT)
