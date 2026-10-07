"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Build a quantizer from the keys that index ``WEIGHT_QUANT_MAP`` / ``INPUT_QUANT_MAP``.

:func:`create_weight_quantizer` and :func:`create_input_quantizer` are drop-in
replacements for the static lookups::

    WEIGHT_QUANT_MAP[format][scale_precision][param_method][granularity][quant_type]
    INPUT_QUANT_MAP[format][scale_type][scale_precision][param_method][granularity][quant_type]
    INPUT_QUANT_MAP[format][scale_type][quant_type]  # scale_type == 'no_scale'

They take the same strings and assemble the equivalent injector through
:class:`~.weight.WeightQuantizerBuilder` / :class:`~.input.InputQuantizerBuilder`
instead of indexing the tables. The result matches the map leaf *before*
finalization, so the ``.let(...)`` chain that ``generate_quantizers`` applies
afterwards (bit width, group size, narrow range, zero-point storage, scale
rounding, signed scale, per-row / per-group runtime attributes, ...) works
unchanged.

Axis values that are not part of the leaf identity are therefore left at their
builder defaults and expected to be supplied by that finalization step.
"""
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.stats_op import AbsMinMax
from brevitas.core.stats.stats_op import NegativeMinOrZero
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder.builder import create_quantizer_builder
from brevitas_examples.common.quantizer_builder.input import InputQuantizerBuilder
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType
from brevitas_examples.common.quantizer_builder.mixins import ZeroPointImplType
from brevitas_examples.common.quantizer_builder.weight import WeightQuantizerBuilder

# Every float leaf of either map is an Fp8e4m3 quantizer. A different eXmY
# requested on the command line is not part of the map key (it is stripped by
# ``quant_format_from_string``) and is applied downstream as explicit
# exponent/mantissa bit widths, which override the ones derived here.
_FLOAT_QUANT_FORMAT = 'e4m3'


def _format_axes(quant_format: str) -> Tuple[QuantType, Optional[FloatFormat]]:
    """'int' -> (INT, None); 'float' / 'float_ocp' / 'float_fnuz' -> (FP, FLOAT / OCP / FNUZ)."""
    if quant_format == 'int':
        return QuantType.INT, None
    return QuantType.FP, FloatFormat[quant_format.rpartition('_')[2].upper()]


def _restrict_scaling_type(scale_precision: str) -> RestrictValueType:
    """'float_scale' -> FP, 'po2_scale' -> POWER_OF_TWO."""
    if scale_precision not in ('float_scale', 'po2_scale'):
        raise ValueError(f"Unknown scale precision '{scale_precision}'.")
    return (
        RestrictValueType.POWER_OF_TWO if scale_precision == 'po2_scale' else RestrictValueType.FP)


def _granularity(quant_granularity: str) -> ScalingPerOutputType:
    """'per_tensor' -> TENSOR, 'per_channel' / 'per_row' -> CHANNEL, 'per_group' -> GROUP."""
    if quant_granularity == 'per_row':
        return ScalingPerOutputType.CHANNEL
    return ScalingPerOutputType[quant_granularity.removeprefix('per_').upper()]


def _weight_split_param_method(
        param_method: ParamMethod,
        quant_param_type: QuantParamType) -> Tuple[ParamMethod, Optional[ParamMethod]]:
    """Spread the requested local-loss method over the scale and the zero-point.

    Mirrors how the WEIGHT_QUANT_MAP leaves compose their local-loss mixins:
      * symmetric: only the scale is searched (MSE/HQO SymmetricScale);
      * asymmetric HQO: *only* the zero-point is searched (HQOWeightZeroPoint);
        the scale stays a plain MinMax stats scale;
      * asymmetric MSE: the zero-point is searched as well (MSEWeightZeroPoint).
    """
    if quant_param_type != QuantParamType.ASYM:
        return param_method, None
    if param_method == ParamMethod.HQO:
        return ParamMethod.STATS, ParamMethod.HQO
    if param_method == ParamMethod.MSE:
        return ParamMethod.MSE, ParamMethod.MSE
    return param_method, None


def _weight_scaling_impl_type(
        param_method: ParamMethod, quant_param_type: QuantParamType) -> ScalingImplType:
    """Scale storage implied by the leaf: the local-loss quantizers keep the
    searched scale as a standalone parameter, except asymmetric HQO, whose scale
    stays a plain stats scale (only its zero-point is optimized)."""
    if param_method == ParamMethod.MSE:
        return ScalingImplType.PARAMETER_FROM_STATS
    if param_method == ParamMethod.HQO and quant_param_type == QuantParamType.SYM:
        return ScalingImplType.PARAMETER_FROM_STATS
    return ScalingImplType.STATS


def _weight_attr_overrides(
        weight_quant_format: str,
        weight_scale_precision: str,
        weight_param_method: str,
        weight_quant_granularity: str,
        weight_quant_type: str) -> Dict[str, Any]:
    """Namespace attributes the reference leaves carry that are not implied by
    the quantization axes themselves."""
    overrides: Dict[str, Any] = {}
    is_group = weight_quant_granularity == 'per_group'
    is_asym = weight_quant_type == 'asym'
    # MX int (MXInt8Weight) is built on IntQuant, which is not narrow, unlike the
    # builder's narrow-by-default symmetric int weights.
    if (weight_quant_format == 'int' and weight_scale_precision == 'po2_scale' and is_group and
            not is_asym):
        overrides['narrow_range'] = False
    # ShiftedUint8Weight...HQO does not quantize its zero-point and stores it as a
    # standalone parameter. The scale stays a stats scale, so that storage cannot
    # be mirrored from scaling_impl_type and has to be requested explicitly.
    if is_asym and weight_param_method == 'hqo':
        overrides['quantize_zero_point'] = False
        overrides['zero_point_impl_type'] = ZeroPointImplType.PARAMETER_FROM_STATS
    return overrides


def create_weight_quantizer(
        weight_quant_format: str,
        weight_scale_precision: str,
        weight_param_method: str,
        weight_quant_granularity: str,
        weight_quant_type: str) -> Type:
    """Build the weight quantizer injector identified by the ``WEIGHT_QUANT_MAP`` keys.

    Args:
        weight_quant_format: 'int' | 'float' | 'float_ocp' | 'float_fnuz'.
        weight_scale_precision: 'float_scale' | 'po2_scale'.
        weight_param_method: 'stats' | 'mse' | 'hqo'.
        weight_quant_granularity: 'per_tensor' | 'per_channel' | 'per_group'.
        weight_quant_type: 'sym' | 'asym'.

    Unlike the table, every combination of the five axes is buildable here: only
    the combinations rejected by :class:`QuantizerConfig` raise.
    """
    quant_type, float_format = _format_axes(weight_quant_format)
    restrict_scaling_type = _restrict_scaling_type(weight_scale_precision)
    granularity = _granularity(weight_quant_granularity)
    param_method = ParamMethod[weight_param_method.upper()]
    quant_param_type = QuantParamType[weight_quant_type.upper()]

    scaling_param_method, zero_point_param_method = _weight_split_param_method(
        param_method, quant_param_type)

    return create_quantizer_builder(
        WeightQuantizerBuilder,
        quant_type,
        quant_param_type=quant_param_type,
        scaling_impl_type=_weight_scaling_impl_type(param_method, quant_param_type),
        scaling_per_output_type=granularity,
        restrict_scaling_type=restrict_scaling_type,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        float_format=float_format,
        float_quant_format=None if float_format is None else _FLOAT_QUANT_FORMAT,
        attr_overrides=_weight_attr_overrides(
            weight_quant_format,
            weight_scale_precision,
            weight_param_method,
            weight_quant_granularity,
            weight_quant_type)).build_quant_injector()


def _input_scaling_impl_type(input_scale_type: str) -> Optional[ScalingImplType]:
    """'static' -> PARAMETER_FROM_STATS, 'dynamic' -> DYNAMIC, 'no_scale' -> None."""
    if input_scale_type not in ('static', 'dynamic', 'no_scale'):
        raise ValueError(f"Unknown input_scale_type '{input_scale_type}'.")
    if input_scale_type == 'no_scale':
        return None
    return (
        ScalingImplType.DYNAMIC
        if input_scale_type == 'dynamic' else ScalingImplType.PARAMETER_FROM_STATS)


def _input_attr_overrides(
        input_scale_type: str, input_param_method: Optional[str],
        input_quant_type: str) -> Dict[str, Any]:
    """Namespace attributes the reference leaves carry that are not implied by
    the quantization axes themselves.

    The static mixin sets ``scaling_stats_op=PERCENTILE``, from which the builder
    would derive an ``AbsPercentile`` local-loss init op, while the reference
    MSESymmetricScale / MSEAsymmetricScale hardcode AbsMax / AbsMinMax (and
    MSEActZeroPoint hardcodes NegativeMinOrZero). MSE is static-only in
    INPUT_QUANT_MAP, so this is the only override the input leaves need.
    """
    if input_scale_type != 'static' or input_param_method != 'mse':
        return {}
    if input_quant_type == 'asym':
        return {'scaling_mse_init_op': AbsMinMax, 'zero_point_mse_init_op': NegativeMinOrZero}
    return {'scaling_mse_init_op': AbsMax}


def create_input_quantizer(
        input_quant_format: str,
        input_scale_type: str,
        input_quant_type: str,
        *,
        input_scale_precision: Optional[str] = None,
        input_param_method: Optional[str] = None,
        input_quant_granularity: Optional[str] = None) -> Type:
    """Build the input quantizer injector identified by the ``INPUT_QUANT_MAP`` keys.

    ``INPUT_QUANT_MAP`` is indexed on six axes, except for ``no_scale`` which is
    indexed on three; the three required arguments here are exactly those three
    keys, and the remaining axes are keyword-only and required for every other
    ``input_scale_type``.

    Args:
        input_quant_format: 'int' | 'float' | 'float_ocp' | 'float_fnuz'.
        input_scale_type: 'static' | 'dynamic' | 'no_scale'.
        input_quant_type: 'sym' | 'asym'.
        input_scale_precision: 'float_scale' | 'po2_scale'; unused for 'no_scale'.
        input_param_method: 'stats' | 'mse'; unused for 'no_scale'.
        input_quant_granularity: 'per_tensor' | 'per_row' | 'per_group'; unused
            for 'no_scale'.

    Unlike the table, every combination of the axes is buildable here: only the
    combinations rejected by :class:`QuantizerConfig` raise.
    """
    quant_type, float_format = _format_axes(input_quant_format)
    quant_param_type = QuantParamType[input_quant_type.upper()]
    scaling_impl_type = _input_scaling_impl_type(input_scale_type)

    if input_scale_type == 'no_scale':
        # The no_scale leaves carry no scale at all (FloatActBase): the remaining
        # axes are not part of their identity, so they keep the builder defaults.
        restrict_scaling_type = RestrictValueType.FP
        granularity = ScalingPerOutputType.TENSOR
        param_method = ParamMethod.STATS
    else:
        missing = [
            name for name,
            value in (('input_scale_precision',
                       input_scale_precision), ('input_param_method', input_param_method),
                      ('input_quant_granularity', input_quant_granularity)) if value is None]
        if missing:
            raise ValueError(
                f"input_scale_type '{input_scale_type}' requires {', '.join(missing)}.")
        restrict_scaling_type = _restrict_scaling_type(input_scale_precision)
        granularity = _granularity(input_quant_granularity)
        param_method = ParamMethod[input_param_method.upper()]

    # Only the asymmetric MSE leaves search the zero-point as well (MSEActZeroPoint).
    zero_point_param_method = (
        ParamMethod.MSE
        if quant_param_type == QuantParamType.ASYM and param_method == ParamMethod.MSE else None)

    return create_quantizer_builder(
        InputQuantizerBuilder,
        quant_type,
        quant_param_type=quant_param_type,
        scaling_impl_type=scaling_impl_type,
        scaling_per_output_type=granularity,
        restrict_scaling_type=restrict_scaling_type,
        scaling_param_method=param_method,
        zero_point_param_method=zero_point_param_method,
        float_format=float_format,
        float_quant_format=None if float_format is None else _FLOAT_QUANT_FORMAT,
        attr_overrides=_input_attr_overrides(
            input_scale_type, input_param_method, input_quant_type)).build_quant_injector()
