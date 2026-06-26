"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
from enum import auto
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
import warnings

from dependencies import this
from dependencies import value
from torch import nn

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.function_wrapper.ops_ste import FloorSte
from brevitas.core.function_wrapper.shape import StatsInputViewShapeImpl
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.stats import MSE
from brevitas.core.stats.stats_op import HalfQuadraticOptimizerScale
from brevitas.core.stats.stats_op import HalfQuadraticOptimizerZeroPoint
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.core.zero_point import ParameterZeroPoint
from brevitas.core.zero_point import StatsFromParameterZeroPoint
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.quant.solver.common import solve_stats_impl
from brevitas.quant.solver.common import SolveBitWidthImplFromEnum
from brevitas.quant.solver.common import SolveIntScalingImplFromEnum
from brevitas.quant.solver.common import SolveRestrictScalingImplFromEnum
from brevitas.quant.solver.common import SolveScalingStatsInputViewShapeImplFromEnum
from brevitas.quant.solver.common import SolveScalingStatsOpFromEnum
from brevitas.quant.solver.common import SolveStatsReduceDimFromEnum
from brevitas.quant.solver.common import SolveTensorQuantFloatToIntImplFromEnum
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.utils.float_quant_utils import get_midmax_mantissa_bit_bias
from brevitas.utils.python_utils import AutoName
from brevitas_examples.common.generative.quantizers import BaseQuantizer

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


class AsymMixin(ExtendedInjector):
    scaling_stats_op = StatsOp.MIN_MAX
    zero_point_stats_op = StatsOp.NEG_MIN_OR_ZERO
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
    zero_point_stats_input_concat_dim = this.scaling_stats_input_concat_dim

    @value
    def zero_point_stats_impl(zero_point_stats_op=None):
        return solve_stats_impl(zero_point_stats_op)

    # The zero-point implementation is driven by the *zero-point* param method
    # (zero_point_impl_type), independent of the scale's scaling_impl_type. When
    # no dedicated zero-point param method is selected (zero_point_impl_type is
    # None), the default asymmetric zero-point is a plain stats-from-parameter
    # zero-point, regardless of whether the scale is STATS/MSE/HQO.
    @value
    def zero_point_impl(
            zero_point_impl_type: EnumType[ScalingImplType] = None) -> Optional[Type[nn.Module]]:
        if zero_point_impl_type is None or zero_point_impl_type == ScalingImplType.STATS:
            return StatsFromParameterZeroPoint
        elif zero_point_impl_type == ScalingImplType.PARAMETER:
            return ParameterZeroPoint
        elif zero_point_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            return ParameterFromStatsFromParameterZeroPoint
        elif zero_point_impl_type == ScalingImplType.AFFINE_STATS:
            return StatsFromParameterZeroPoint
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
    restrict_value_float_to_int_impl = FloorSte

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


@value
def scale_init_op(
    scaling_stats_op: EnumType[StatsOp] = None,
    restrict_scaling_type: EnumType[RestrictValueType] = None,
) -> Optional[nn.Module]:
    return solve_stats_impl(scaling_stats_op, restrict_scaling_type)


@value
def zero_point_init_op(zero_point_stats_op: EnumType[StatsOp] = None,) -> Optional[nn.Module]:
    return solve_stats_impl(zero_point_stats_op)


def _make_mse_injector(target: Target = Target.SCALE) -> Type[ExtendedInjector]:
    MSESubInjector = type(
        f'MSE{target.capitalize()}SubInjector', (MSESubInjectorMixin,), {
            "mse_init_op": getattr(this << 1, f"{target}_mse_init_op"),})
    prefix = {
        Target.SCALE.value: "scaling",
        Target.ZERO_POINT.value: "zero_point",}[target.value]

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
        f"{target}_mse_init_op": _make_scaling_init_op(prefix),
        "inner_stats_input_view_shape_impl": inner_stats_input_view_shape_impl,}
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


# Scale sub-injector: init op is derived at the *parent* (MSEInjectorMixin)
# level from the requested enums, and pulled in here via `(this << 1)`. The
# zero-point sub-injector (added later) will instead use a fixed init op.
MSEScaleInjectorMixin = _make_mse_injector(target=Target.SCALE)
MSEZeroPointInjectorMixin = _make_mse_injector(target=Target.ZERO_POINT)

HQOScaleInjectorMixin = _make_hqo_injector(target=Target.SCALE)
HQOZeroPointInjectorMixin = _make_hqo_injector(target=Target.ZERO_POINT)


class BaseQuantSolver(SolveStatsReduceDimFromEnum,
                      SolveScalingStatsInputViewShapeImplFromEnum,
                      SolveScalingStatsOpFromEnum,
                      SolveBitWidthImplFromEnum,
                      SolveTensorQuantFloatToIntImplFromEnum,
                      SolveRestrictScalingImplFromEnum,
                      SolveIntScalingImplFromEnum):
    """Common solver mixins shared by activation, weight and bias quantizers.

    Brevitas' ``ActQuantSolver``, ``WeightQuantSolver`` and ``BiasQuantSolver``
    each inherit a large list of ``Solve*FromEnum`` mixins that translate enum
    directives into concrete quantization core modules. The mixins gathered
    here are exactly the ones common to all three solvers, so they can be
    reused as a generic base regardless of what is being quantized (weights or
    activations). Quantizer-kind-specific mixins (parameter vs. activation
    handling, tensor-quant resolution, scaling shapes, ...) are intentionally
    left out and must be added by the concrete builder/quantizer.
    """
    # TODO: Add Solve for QuantType
    pass


"""
TODOs: The following docstring documents findings when investigating the way to
implement a generic quantizer builder.

- The way QuantType is resolved differs for weights and activations: for activations
  BINARY is resolved to ClampedBinaryQuant, instead of to BinaryQuant.
"""


class QuantizerBuilder:
    """Builds Brevitas quantizer injectors programmatically.

    This is intended to eventually replace the static ``WEIGHT_QUANT_MAP`` and
    ``INPUT_QUANT_MAP`` lookup tables in
    ``brevitas_examples.common.generative.quantize`` with explicit
    construction logic.

    The builder is deliberately generic: it describes a quantizer in terms of
    its quantization axes (format, scale precision, param method, granularity,
    quant type, ...) without committing to whether it quantizes weights or
    activations, so the same infrastructure can be shared by both.

    For now this is a minimal stub: the public API is fixed so tests can be
    written against it, but the actual building logic is not implemented yet.
    """

    def __init__(
        self,
        quant_type: Union[str, QuantType],
        quant_param_type: Union[str, QuantParamType] = QuantParamType.SYM,
        # General quantization parameters
        bit_width: int = 8,
        bit_width_impl_type: Union[str, BitWidthImplType] = BitWidthImplType.CONST,
        float_to_int_impl_type: Union[str, FloatToIntImplType] = FloatToIntImplType.ROUND,
        # Scaling parameters
        scaling_impl_type: Union[str, ScalingImplType] = ScalingImplType.STATS,
        scaling_per_output_type: Union[str, ScalingPerOutputType] = ScalingPerOutputType.TENSOR,
        restrict_scaling_type: Union[str, RestrictValueType] = RestrictValueType.FP,
        scaling_min_val: Optional[float] = None,
        scaling_param_method: Union[str, ParamMethod] = ParamMethod.STATS,
        # Zero point parameters
        zero_point_param_method: Optional[Union[str, ParamMethod]] = None,
        # Additional kwargs to be injected into the quantizer injector
        kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.quant_type = quant_type
        self.quant_param_type = quant_param_type
        self.bit_width = bit_width
        self.bit_width_impl_type = bit_width_impl_type
        self.float_to_int_impl_type = float_to_int_impl_type
        self.scaling_impl_type = scaling_impl_type
        self.scaling_per_output_type = scaling_per_output_type
        self.restrict_scaling_type = restrict_scaling_type
        self.scaling_min_val = scaling_min_val
        self.scaling_param_method = scaling_param_method
        self.zero_point_param_method = zero_point_param_method
        self.kwargs = kwargs

    def build_quant_injector(
            self,
            quant_injector: Type[ExtendedInjector] = ExtendedInjector,
            value_solve_fns: Optional[List[Callable]] = None) -> Any:
        """Return the customized quantizer injector.

        The result must be equivalent to the corresponding quantizer entry
        produced by
        ``brevitas_examples.common.generative.quantize.generate_quantizers``
        for the same set of quantization arguments.
        """
        namespace: Dict[str, Any] = {}

        # Insert quant_type into the injector
        namespace['quant_type'] = self.quant_type
        namespace['bit_width'] = self.bit_width
        namespace['bit_width_impl_type'] = self.bit_width_impl_type
        namespace['float_to_int_impl_type'] = self.float_to_int_impl_type

        namespace['scaling_impl_type'] = self.scaling_impl_type
        namespace['scaling_per_output_type'] = self.scaling_per_output_type
        namespace['restrict_scaling_type'] = self.restrict_scaling_type
        namespace['scaling_min_val'] = self.scaling_min_val

        # Relevant parameters depending on the type of quantization (symmetric vs asymmetric)
        if self.quant_param_type == QuantParamType.SYM:
            if self.quant_type in [QuantType.INT]:
                namespace['signed'] = True
                namespace['narrow_range'] = True
        elif self.quant_param_type == QuantParamType.ASYM:
            if self.quant_type in [QuantType.INT]:
                namespace['signed'] = False
                namespace['narrow_range'] = False
                namespace['quantize_zero_point'] = True
        else:
            raise RuntimeError(f"Unrecognized quant_param_type {self.quant_param_type}")

        # The MSE init op (e.g. AbsMax) must reduce *without* keepdim so its output
        # shape matches the per-group MSE loss (e.g. (out, n_groups)) during the
        # grid search. Groupwise quantizers set keepdim=True for the standalone
        # scaling stats, but brevitas' MSE sub-injectors instantiate the init op
        # with keepdim=False; we mirror that here to avoid a shape mismatch in
        # mse_grid_search for groupwise (MX) MSE quantizers.
        if (self.scaling_param_method == ParamMethod.MSE):
            namespace['keepdim'] = False

        # Insert value_solve_fns into the injector if provided
        if value_solve_fns:
            for i, fn in enumerate(value_solve_fns):
                fn_name = getattr(fn, '__name__', f'value_solve_fn_{i}')
                namespace[fn_name] = value(fn)

        # Add additional kwargs
        namespace.update(self.kwargs or {})

        base_classes = (WeightQuantSolver,)

        # Handle restrict_scaling_type
        if self.restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            if self.scaling_per_output_type == ScalingPerOutputType.GROUP:
                base_classes += (GroupwisePoTMixin,)
            else:
                # TODO (pml): Switch to FloatToIntImplType.FLOOR_STE when adding
                # the solve function
                namespace['restrict_value_float_to_int_impl'] = CeilSte

        # If zero point is enabled, the appropiate asym mixin is provided
        if self.quant_param_type == QuantParamType.ASYM:
            base_classes += (AsymMixin,)
        elif self.quant_param_type == QuantParamType.SYM:
            base_classes += (SymMixin,)

        if self.scaling_param_method == ParamMethod.MSE:
            base_classes = (MSEScaleInjectorMixin,) + base_classes
            # Force the scaling_impl_type to be PARAMETER_FROM_STATS when using MSE
            del namespace['scaling_impl_type']
        elif self.scaling_param_method == ParamMethod.HQO:
            base_classes = (HQOScaleInjectorMixin,) + base_classes
            # Force the scaling_impl_type to be PARAMETER_FROM_STATS when using HQO
            del namespace['scaling_impl_type']

        if self.zero_point_param_method is not None and self.quant_param_type == QuantParamType.SYM:
            raise ValueError(
                "Zero point parameter method is not applicable for symmetric quantization.")

        if self.zero_point_param_method == ParamMethod.MSE:
            base_classes = (MSEZeroPointInjectorMixin,) + base_classes
        elif self.zero_point_param_method == ParamMethod.HQO:
            base_classes = (HQOZeroPointInjectorMixin,) + base_classes

        return type("QuantInjector", base_classes, namespace)
