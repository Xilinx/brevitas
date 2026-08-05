"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
from abc import ABC
from abc import abstractmethod
from enum import auto
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
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
from brevitas.core.zero_point import ParameterFromRuntimeZeroPoint
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
from brevitas.quant.float_quant_fnuz import FpFNUZMixin
from brevitas.quant.float_quant_ocp import FpOCPMixin
from brevitas.quant.solver.common import solve_float_to_int_impl_from_enum
from brevitas.quant.solver.common import solve_stats_impl
from brevitas.quant.solver.common import SolveBitWidthImplFromEnum
from brevitas.quant.solver.common import SolveIntScalingImplFromEnum
from brevitas.quant.solver.common import SolveRestrictScalingImplFromEnum
from brevitas.quant.solver.common import SolveScalingStatsInputViewShapeImplFromEnum
from brevitas.quant.solver.common import SolveScalingStatsOpFromEnum
from brevitas.quant.solver.common import SolveStatsReduceDimFromEnum
from brevitas.quant.solver.common import SolveTensorQuantFloatToIntImplFromEnum
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
    # (zero_point_impl_type), independent of the scale's scaling_impl_type. When
    # no dedicated zero-point param method is selected (zero_point_impl_type is
    # None), the default asymmetric zero-point is a plain stats-from-parameter
    # zero-point, regardless of whether the scale is STATS/MSE/HQO.
    @value
    def zero_point_impl(
            zero_point_impl_type: EnumType[ZeroPointImplType] = None) -> Optional[Type[nn.Module]]:
        if zero_point_impl_type is None or zero_point_impl_type == ZeroPointImplType.STATS:
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


class ScaleType(AutoName):
    # Input/activation-only axis (maps to INPUT_QUANT_MAP keys).
    STATIC = auto()  # scale computed from runtime stats, stored as a parameter
    DYNAMIC = auto()  # scale recomputed per-forward
    NO_SCALE = auto()  # no scale (float only, e.g. Fp8e4m3Act)


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


class BaseQuantizerBuilder(ABC):
    """Builds Brevitas quantizer injectors programmatically.

    This is intended to eventually replace the static ``WEIGHT_QUANT_MAP`` and
    ``INPUT_QUANT_MAP`` lookup tables in
    ``brevitas_examples.common.generative.quantize`` with explicit
    construction logic.

    The builder is deliberately generic: it describes a quantizer in terms of
    its quantization axes (format, scale precision, param method, granularity,
    quant type, ...) without committing to whether it quantizes weights or
    activations, so the same infrastructure can be shared by both.

    Convention: every ``_build_*`` override calls ``super()`` **first**, then
    applies its own ``namespace`` / ``base_classes`` modifications. This way the
    most-derived builder always has the final say over any ``namespace`` key,
    making override precedence easy to reason about.

    For now this is a minimal stub: the public API is fixed so tests can be
    written against it, but the actual building logic is not implemented yet.
    """

    def __init__(
        self,
        quant_param_type: Union[str, QuantParamType] = QuantParamType.SYM,
        # General quantization parameters
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
        self.quant_param_type = quant_param_type
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
            base_classes: Optional[Tuple[Type, ...]] = None,
            value_solve_fns: Optional[List[Callable]] = None) -> Any:
        """Return the customized quantizer injector.

        The result must be equivalent to the corresponding quantizer entry
        produced by
        ``brevitas_examples.common.generative.quantize.generate_quantizers``
        for the same set of quantization arguments.
        """
        if base_classes is None:
            base_classes = tuple()

        # Append the solver / quantizer base classes (weight vs activation,
        # int vs float). Concrete builders provide these via a hook so the
        # input builders can cooperatively swap weight bases for activation ones.
        base_classes = base_classes + self._solver_base_classes()

        namespace: Dict[str, Any] = self._build_base_namespace()
        # The proxy class is provided by the concrete builder (weight vs activation)
        namespace['proxy_class'] = self._proxy_class()

        # Enable the scaling parameter method (MSE / HQO) if specified
        base_classes = self._build_scaling_param_method(namespace, base_classes)
        # Build the restrict scaling method if specified
        base_classes = self._build_restrict_param_method(namespace, base_classes)

        # If zero point is enabled, the appropiate asym mixin is provided
        if self.quant_param_type == QuantParamType.ASYM:
            base_classes = self._build_asymmmetric_quantizer(namespace, base_classes)
        elif self.quant_param_type == QuantParamType.SYM:
            base_classes = self._build_symmmetric_quantizer(namespace, base_classes)

        # Insert value_solve_fns into the injector if provided
        if value_solve_fns:
            for i, fn in enumerate(value_solve_fns):
                fn_name = getattr(fn, '__name__', f'value_solve_fn_{i}')
                namespace[fn_name] = value(fn)

        # Add additional kwargs
        namespace.update(self.kwargs or {})

        return type("QuantInjector", base_classes, namespace)

    def describe_quantizer(self, resolve: bool = True) -> None:
        """Build the quant injector and print its attributes, dependency kinds,
        and (for ``@value`` functions) the args they require and resolve to."""
        from brevitas_examples.common.injector_utils import describe_injector
        describe_injector(self.build_quant_injector(), resolve=resolve)

    def _build_base_namespace(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {}
        namespace['bit_width_impl_type'] = self.bit_width_impl_type
        namespace['float_to_int_impl_type'] = self.float_to_int_impl_type

        namespace['scaling_impl_type'] = self.scaling_impl_type
        namespace['scaling_per_output_type'] = self.scaling_per_output_type
        namespace['restrict_scaling_type'] = self.restrict_scaling_type
        namespace['scaling_min_val'] = self.scaling_min_val
        return namespace

    def _build_scaling_param_method(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        """Prepend the scale param-method mixin (MSE / HQO) and adjust namespace."""
        # The MSE init op (e.g. AbsMax) must reduce *without* keepdim so its
        # output shape matches the per-group MSE loss (e.g. (out, n_groups))
        # during the grid search. Groupwise quantizers set keepdim=True for the
        # standalone scaling stats, but brevitas' MSE sub-injectors instantiate
        # the init op with keepdim=False; we mirror that here to avoid a shape
        # mismatch in mse_grid_search for groupwise (MX) MSE quantizers.
        if self.scaling_param_method == ParamMethod.MSE:
            base_classes = (MSEScaleInjectorMixin,) + base_classes
            # Force the scaling_impl_type to be PARAMETER_FROM_STATS when using MSE/HQO
            namespace.pop('scaling_impl_type', None)
        elif self.scaling_param_method == ParamMethod.HQO:
            base_classes = (HQOScaleInjectorMixin,) + base_classes
            # Force the scaling_impl_type to be PARAMETER_FROM_STATS when using MSE/HQO
            namespace.pop('scaling_impl_type', None)
        return base_classes

    def _build_zero_point_param_method(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        """Prepend the zero_point param-method mixin (MSE / HQO) and adjust namespace."""
        if self.zero_point_param_method == ParamMethod.MSE:
            base_classes = (MSEZeroPointInjectorMixin,) + base_classes
        elif self.zero_point_param_method == ParamMethod.HQO:
            base_classes = (HQOZeroPointInjectorMixin,) + base_classes

        return base_classes

    def _build_symmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        if self.zero_point_param_method is not None and self.quant_param_type == QuantParamType.SYM:
            raise ValueError(
                "Zero point parameter method is not applicable for symmetric quantization.")
        base_classes += (SymMixin,)
        return base_classes

    def _build_asymmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        # Enable asymmetric quantization by adding the AsymmetricZeroPointMixin
        base_classes += (AsymmetricZeroPointMixin,)
        base_classes = self._build_zero_point_param_method(namespace, base_classes)
        return base_classes

    def _build_restrict_param_method(
        self,
        namespace: Dict[str, Any],
        base_classes: Tuple[Type, ...],
        restrict_value_float_to_int_impl_type: EnumType[FloatToIntImplType] = FloatToIntImplType
        .CEIL
    ) -> Tuple[Type, ...]:
        # Specify the float-to-int implementation for restricting the scaling value to an integer.
        # Non-group power-of-two scaling ceils the exponent (PerTensor/PerChannelPoTScaling8bit,
        # Int8ActPerTensorFixedPoint); groupwise (MX) power-of-two scaling floors it (MXMixin).
        if self.restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            if self.scaling_per_output_type == ScalingPerOutputType.GROUP:
                base_classes += (GroupwisePoTMixin,)
                restrict_value_float_to_int_impl_type = FloatToIntImplType.FLOOR

            # TODO (pml): Add @value for solve_float_to_int_impl_from_enum to appropiate classes
            namespace['restrict_value_float_to_int_impl'] = solve_float_to_int_impl_from_enum(
                restrict_value_float_to_int_impl_type)
        return base_classes

    @abstractmethod
    def _proxy_class(self) -> Type[nn.Module]:
        pass

    @abstractmethod
    def _solver_base_classes(self) -> Tuple[Type, ...]:
        """Return the solver / quantizer base classes (e.g. WeightQuantSolver)."""
        pass


# ----------------------------------------------------------------------
# Format axis: int vs float. The solver / proxy are deferred to kind-specific
# hooks (``_quant_solver`` / ``_proxy_class``) implemented by the concrete leaf
# builders (see weight_quantizer_builder / input_quantizer_builder).
# ----------------------------------------------------------------------
class IntQuantizerBuilder(BaseQuantizerBuilder):
    """Format axis: integer quantization."""

    def __init__(self, *, bit_width: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bit_width: int = bit_width

    def _build_base_namespace(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = super()._build_base_namespace()
        namespace['quant_type'] = QuantType.INT
        namespace['bit_width'] = self.bit_width
        return namespace

    def _build_symmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        base_classes = super()._build_symmmetric_quantizer(namespace, base_classes)
        namespace['signed'] = True
        namespace['narrow_range'] = True
        return base_classes

    def _build_asymmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        base_classes = super()._build_asymmmetric_quantizer(namespace, base_classes)
        namespace['signed'] = False
        namespace['narrow_range'] = False
        namespace['quantize_zero_point'] = True
        return base_classes

    def _solver_base_classes(self) -> Tuple[Type, ...]:
        return (self._quant_solver(),)


class FloatQuantizerBuilder(BaseQuantizerBuilder):
    """Format axis: float quantization."""

    def __init__(
        self,
        *,
        float_quant_format: str,
        float_format: Union[str, FloatFormat] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.float_format: Optional[Union[str, FloatFormat]] = float_format
        self.float_quant_format: Optional[str] = float_quant_format

        # TODO (pml): Remove or refactor
        if self.quant_param_type != QuantParamType.SYM:
            raise ValueError("Float quantizers only support symmetric quantization.")
        if self.zero_point_param_method is not None:
            raise ValueError("Float quantizers do not support a zero-point param method.")
        # Power-of-two scaled groupwise float (MX) is only defined for the OCP
        # format: it relies on FpOCPMixin's inf/nan values to compute
        # midmax_mantissa_bit_bias.
        # TODO (pml): Double check if this should be the case
        if (self.restrict_scaling_type == RestrictValueType.POWER_OF_TWO and
                self.scaling_per_output_type == ScalingPerOutputType.GROUP and
                FloatFormat(self.float_format) != FloatFormat.OCP):
            raise ValueError(
                "Groupwise power-of-two scaled float quantizers (MX) are only "
                "supported for FloatFormat.OCP.")

    def _solver_base_classes(self) -> Tuple[Type, ...]:
        # The format mixin (OCP / FNUZ) is composed first so its overrides
        # (inf/nan values, exponent_bias) take precedence over the float base.
        base_classes: Tuple[Type, ...] = ()
        format_mixin = FLOAT_FORMAT_MIXIN_MAP.get(FloatFormat(self.float_format).value)
        if format_mixin is not None:
            base_classes = (format_mixin,) + base_classes
        return base_classes

    def _build_base_namespace(self) -> Dict[str, Any]:
        exponent_bit_width, mantissa_bit_width, bit_width = parse_float_quant_format(
            self.float_quant_format)

        namespace: Dict[str, Any] = super()._build_base_namespace()

        namespace['quant_type'] = QuantType.FP
        namespace['bit_width'] = bit_width
        namespace['exponent_bit_width'] = exponent_bit_width
        namespace['mantissa_bit_width'] = mantissa_bit_width
        # All FloatFormat mixins set saturating=True; set it for the plain FLOAT
        # format too (the reference Fp8e4m3Mixin sets it).
        namespace['saturating'] = True

        return namespace
