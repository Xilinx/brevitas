"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Lean, enum-driven quantizer generation built on top of the
:mod:`quantizer_builder` package.

This is a leaner take on ``brevitas_examples.common.generative.quantize.
generate_quantizers``: instead of indexing the static ``WEIGHT_QUANT_MAP`` /
``INPUT_QUANT_MAP`` tables and threading string keys around, the injectors are
assembled directly from their quantization axes via ``build_weight_quantizer`` /
``build_input_quantizer``. The two concerns are split into two functions --
:func:`generate_weight_quantizer` and :func:`generate_input_quantizers` -- and
every argument is a brevitas / builder enum rather than a string.

Only genuinely *layer-supplied* attributes (per-row / per-group broadcast and
reduce dims, group sizes, attention permute dims) are still applied here via
``.let(...)``; they are not part of the quantizer definition.
"""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder import build_input_quantizer
from brevitas_examples.common.quantizer_builder import build_weight_quantizer
from brevitas_examples.common.quantizer_builder import Component
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType

# Runtime, layer-supplied per-row overrides for an input/activation quantizer
# hosted by a linear/conv layer (the layer supplies the reduce/broadcast dims).
_PER_ROW_INPUT_LET = {
    'dynamic_scaling_broadcastable_fn': lambda x,
                                        shape: x.view(*shape[:-1], 1),
    'permute_dims': None,
    'stats_reduce_dim': 1,}


def _scale_rounding_kwargs(scale_rounding_impl: Optional[Type]) -> Dict[str, Any]:
    if scale_rounding_impl is None:
        return {}
    return {'restrict_value_float_to_int_impl': scale_rounding_impl}


def _apply_input_granularity(
        injector, granularity: ScalingPerOutputType, group_size: Optional[int]):
    """Apply the per-row / per-group runtime overrides an input quantizer needs
    when hosted by a linear/conv layer."""
    if injector is None:
        return None
    if granularity == ScalingPerOutputType.CHANNEL:  # per-row
        return injector.let(**_PER_ROW_INPUT_LET)
    if granularity == ScalingPerOutputType.GROUP:
        return injector.let(group_dim=-1, group_size=group_size)
    return injector


def generate_weight_quantizer(
        quant_type: QuantType,
        *,
        quant_param_type: QuantParamType = QuantParamType.SYM,
        param_method: ParamMethod = ParamMethod.STATS,
        granularity: ScalingPerOutputType = ScalingPerOutputType.CHANNEL,
        scale_precision: RestrictValueType = RestrictValueType.FP,
        scaling_impl_type: ScalingImplType = ScalingImplType.PARAMETER_FROM_STATS,
        bit_width: int = 8,
        group_size: Optional[int] = None,
        group_dim: Optional[int] = None,
        quantize_zero_point: bool = False,
        float_format: Optional[FloatFormat] = None,
        float_quant_format: str = 'e4m3',
        scale_rounding_impl: Optional[Type] = None,
        scaling_min_val: Optional[float] = 1e-4,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        extra_components: Optional[List[Component]] = None) -> Type:
    """Build a weight quantizer injector from its quantization axes."""
    kwargs: Dict[str, Any] = {
        'narrow_range': False,
        'quantize_zero_point': quantize_zero_point,}
    kwargs.update(_scale_rounding_kwargs(scale_rounding_impl))
    if group_dim is not None:
        kwargs['group_dim'] = group_dim
    if granularity == ScalingPerOutputType.GROUP and group_size is not None:
        kwargs['group_size'] = group_size
    # Asymmetric parameter-from-stats weights fold the zero-point into a standalone
    # parameter (per_group quantizers already do this by default).
    if (quant_param_type == QuantParamType.ASYM and
            scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS):
        kwargs['zero_point_impl'] = ParameterFromStatsFromParameterZeroPoint
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Split the requested local-loss method across the scale and the zero-point,
    # mirroring the WEIGHT_QUANT_MAP reference quantizers:
    #   sym        -> scale uses the method, no zero-point method
    #   asym + MSE -> both scale and zero-point are learned via MSE
    #   asym + HQO -> only the zero-point is learned via HQO; the scale stays a
    #                 plain MinMax stats scale (matches ShiftedUint8Weight...HQO)
    scaling_param_method = param_method
    zero_point_param_method = None
    if quant_param_type == QuantParamType.ASYM:
        if param_method == ParamMethod.HQO:
            scaling_param_method = ParamMethod.STATS
            zero_point_param_method = ParamMethod.HQO
        elif param_method == ParamMethod.MSE:
            zero_point_param_method = ParamMethod.MSE

    return build_weight_quantizer(
        quant_type,
        quant_param_type=quant_param_type,
        bit_width=bit_width,
        scaling_impl_type=scaling_impl_type,
        scaling_per_output_type=granularity,
        restrict_scaling_type=scale_precision,
        scaling_min_val=scaling_min_val,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        float_format=float_format,
        float_quant_format=float_quant_format,
        extra_components=extra_components,
        kwargs=kwargs).build_quant_injector()


def _build_input_quant(
        quant_type: QuantType,
        quant_param_type: QuantParamType,
        scale_type: ScaleType,
        granularity: ScalingPerOutputType,
        scale_precision: RestrictValueType,
        param_method: ParamMethod,
        bit_width: int,
        quantize_zero_point: bool,
        float_format: Optional[FloatFormat],
        float_quant_format: str,
        scale_rounding_impl: Optional[Type],
        scaling_min_val: Optional[float],
        extra_kwargs: Optional[Dict[str, Any]]) -> Type:
    kwargs: Dict[str, Any] = {'quantize_zero_point': quantize_zero_point}
    kwargs.update(_scale_rounding_kwargs(scale_rounding_impl))
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return build_input_quantizer(
        quant_type,
        quant_param_type=quant_param_type,
        scale_type=scale_type,
        bit_width=bit_width,
        scaling_per_output_type=granularity,
        restrict_scaling_type=scale_precision,
        scaling_min_val=scaling_min_val,
        scaling_param_method=param_method,
        float_format=float_format,
        float_quant_format=float_quant_format,
        kwargs=kwargs).build_quant_injector()


def generate_input_quantizers(
        quant_type: QuantType,
        *,
        scale_type: ScaleType = ScaleType.STATIC,
        quant_param_type: QuantParamType = QuantParamType.SYM,
        param_method: ParamMethod = ParamMethod.STATS,
        granularity: ScalingPerOutputType = ScalingPerOutputType.TENSOR,
        scale_precision: RestrictValueType = RestrictValueType.FP,
        bit_width: int = 8,
        group_size: Optional[int] = None,
        quantize_zero_point: bool = False,
        float_format: Optional[FloatFormat] = None,
        float_quant_format: str = 'e4m3',
        scale_rounding_impl: Optional[Type] = None,
        scaling_min_val: Optional[float] = 1e-4,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        # Attention axes; each falls back to the corresponding input axis when None.
        attn_quant_config: str = "qkvs",  # choices: "kv", "qkvs", "qkv"
        quant_attn_mode: str = 'mha',  # choices: "mha", "sdpa"
        attn_quant_type: Optional[QuantType] = None,
        attn_scale_type: Optional[ScaleType] = None,
        attn_quant_param_type: Optional[QuantParamType] = None,
        attn_param_method: Optional[ParamMethod] = None,
        attn_granularity: Optional[ScalingPerOutputType] = None,
        attn_scale_precision: Optional[RestrictValueType] = None,
        attn_bit_width: Optional[int] = None,
        attn_group_size: Optional[int] = None,
        attn_float_format: Optional[FloatFormat] = None,
        attn_float_quant_format: Optional[str] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Type]:
    """Build the input / activation quantizer injectors from their axes.

    Returns a dict with ``linear_input_quant``, ``input_quant`` and the four
    attention quantizers (``q_scaled_quant``, ``k_transposed_quant``,
    ``v_quant``, ``attn_output_weights_quant``).
    """
    # no_scale is float-only and has no scale-precision / param-method axis.
    if scale_type == ScaleType.NO_SCALE:
        no_scale_quant = _build_input_quant(
            quant_type=quant_type,
            quant_param_type=quant_param_type,
            scale_type=scale_type,
            granularity=granularity,
            scale_precision=scale_precision,
            param_method=param_method,
            bit_width=bit_width,
            quantize_zero_point=quantize_zero_point,
            float_format=float_format,
            float_quant_format=float_quant_format,
            scale_rounding_impl=scale_rounding_impl,
            scaling_min_val=scaling_min_val,
            extra_kwargs=extra_kwargs)
        return {
            'linear_input_quant': no_scale_quant,
            'input_quant': no_scale_quant,
            'q_scaled_quant': None,
            'k_transposed_quant': None,
            'v_quant': None,
            'attn_output_weights_quant': None,}

    # input_quant and linear_input_quant share the same configuration; build two
    # independent injectors so downstream granularity overrides stay separate.
    input_build_args = dict(
        quant_type=quant_type,
        quant_param_type=quant_param_type,
        scale_type=scale_type,
        granularity=granularity,
        scale_precision=scale_precision,
        param_method=param_method,
        bit_width=bit_width,
        quantize_zero_point=quantize_zero_point,
        float_format=float_format,
        float_quant_format=float_quant_format,
        scale_rounding_impl=scale_rounding_impl,
        scaling_min_val=scaling_min_val,
        extra_kwargs=extra_kwargs)
    input_quant = _build_input_quant(**input_build_args)
    linear_input_quant = _build_input_quant(**input_build_args)

    input_quant = _apply_input_granularity(input_quant, granularity, group_size)
    linear_input_quant = _apply_input_granularity(linear_input_quant, granularity, group_size)

    attn_quants = _generate_attention_quantizers(
        attn_quant_config=attn_quant_config,
        quant_attn_mode=quant_attn_mode,
        quant_type=attn_quant_type if attn_quant_type is not None else quant_type,
        scale_type=attn_scale_type if attn_scale_type is not None else scale_type,
        quant_param_type=(
            attn_quant_param_type if attn_quant_param_type is not None else quant_param_type),
        param_method=attn_param_method if attn_param_method is not None else param_method,
        granularity=attn_granularity if attn_granularity is not None else granularity,
        scale_precision=(
            attn_scale_precision if attn_scale_precision is not None else scale_precision),
        bit_width=attn_bit_width if attn_bit_width is not None else bit_width,
        group_size=attn_group_size if attn_group_size is not None else group_size,
        quantize_zero_point=quantize_zero_point,
        float_format=attn_float_format if attn_float_format is not None else float_format,
        float_quant_format=(
            attn_float_quant_format if attn_float_quant_format is not None else float_quant_format),
        scale_rounding_impl=scale_rounding_impl,
        scaling_min_val=scaling_min_val,
        extra_kwargs=attn_kwargs)

    return {
        'linear_input_quant': linear_input_quant,
        'input_quant': input_quant,
        **attn_quants,}


def _generate_attention_quantizers(
        *,
        attn_quant_config: str,
        quant_attn_mode: str,
        quant_type: QuantType,
        scale_type: ScaleType,
        quant_param_type: QuantParamType,
        param_method: ParamMethod,
        granularity: ScalingPerOutputType,
        scale_precision: RestrictValueType,
        bit_width: int,
        group_size: Optional[int],
        quantize_zero_point: bool,
        float_format: Optional[FloatFormat],
        float_quant_format: str,
        scale_rounding_impl: Optional[Type],
        scaling_min_val: Optional[float],
        extra_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Type]:
    """Build the four attention quantizers (q_scaled / k_transposed / v /
    attn_output_weights) and apply their layer-supplied granularity overrides."""
    k_transposed_quant = _build_input_quant(
        quant_type=quant_type,
        quant_param_type=quant_param_type,
        scale_type=scale_type,
        granularity=granularity,
        scale_precision=scale_precision,
        param_method=param_method,
        bit_width=bit_width,
        quantize_zero_point=quantize_zero_point,
        float_format=float_format,
        float_quant_format=float_quant_format,
        scale_rounding_impl=scale_rounding_impl,
        scaling_min_val=scaling_min_val,
        extra_kwargs=extra_kwargs)

    if attn_quant_config in ("qkvs", "qkv"):
        q_scaled_quant = k_transposed_quant
    elif attn_quant_config == "kv":
        q_scaled_quant = None
    else:
        raise ValueError(
            f"Unknown option for attn_quant_config. attn_quant_config={attn_quant_config}")

    if quant_attn_mode == 'sdpa':
        kv_permute_dims = (0, 1, 3, 2)
        kv_broadcastable_shape_lambda = lambda x, shape: x.view(shape[0], shape[1], 1, shape[-1])
    elif quant_attn_mode == 'mha':
        kv_permute_dims = (0, 2, 1)
        kv_broadcastable_shape_lambda = lambda x, shape: x.view(shape[0], 1, shape[-1])
    else:
        raise ValueError(f"Unknown quant_attn_mode {quant_attn_mode!r}.")

    if granularity == ScalingPerOutputType.CHANNEL:  # per-row
        if q_scaled_quant is not None:
            q_scaled_quant = q_scaled_quant.let(**_PER_ROW_INPUT_LET)
        k_transposed_quant = k_transposed_quant.let(
            dynamic_scaling_broadcastable_fn=kv_broadcastable_shape_lambda,
            permute_dims=kv_permute_dims,
            stats_reduce_dim=1)
    elif granularity == ScalingPerOutputType.GROUP:
        if q_scaled_quant is not None:
            q_scaled_quant = q_scaled_quant.let(group_dim=-1, group_size=group_size)
        k_transposed_quant = k_transposed_quant.let(group_dim=-2, group_size=group_size)
    v_quant = k_transposed_quant

    # If we only quantize QKV, attn_output_weights_quant is left unquantized.
    attn_output_weights_quant = None if attn_quant_config == 'qkv' else q_scaled_quant

    # Attention-output weights are unsigned for symmetric integer attention.
    if (quant_param_type == QuantParamType.SYM and quant_type != QuantType.FP and
            attn_output_weights_quant is not None):
        attn_output_weights_quant = attn_output_weights_quant.let(signed=False)

    return {
        'q_scaled_quant': q_scaled_quant,
        'k_transposed_quant': k_transposed_quant,
        'v_quant': v_quant,
        'attn_output_weights_quant': attn_output_weights_quant,}
