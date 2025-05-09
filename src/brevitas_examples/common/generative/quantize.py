"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
import re

from dependencies import this
import torch
from torch import nn

from brevitas import nn as qnn
from brevitas.core.function_wrapper import CeilSte
from brevitas.core.function_wrapper import FloorSte
from brevitas.core.restrict_val import RoundSte
from brevitas.core.stats import NegativeMinOrZero
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant.base import ParameterFromRuntimeZeroPoint
from brevitas.quant.experimental.float import Fp8e4m3Act
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas.quant.experimental.mx_quant_ocp import MXInt8WeightMSE
from brevitas.quant.experimental.mx_quant_ocp import ShiftedMXUInt8Weight
from brevitas.quant.experimental.mx_quant_ocp import ShiftedMXUInt8WeightMSE
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPointMSE
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerGroupFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.common.generative.nn import LoRACompatibleQuantConv2d
from brevitas_examples.common.generative.nn import LoRACompatibleQuantLinear
from brevitas_examples.common.generative.quantizers import Fp8e4m3DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3FNUZDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3FNUZDynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPDynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightPerChannelFixedPointMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightPerChannelFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import RuntimeDynamicStatsZeroPoint
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat

WEIGHT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat},
                'per_group': {
                    'sym': IntWeightSymmetricGroupQuant,
                    'asym': ShiftedUint8WeightGroupQuantFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatMSE,
                    'asym': ShiftedUint8WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatMSE,
                    'asym': ShiftedUint8WeightPerChannelFloatMSE}},
            'hqo': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatHQO,
                    'asym': ShiftedUint8WeightPerTensorFloatHQO},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatHQO,
                    'asym': ShiftedUint8WeightPerChannelFloatHQO},
                'per_group': {
                    'asym': ShiftedUint8WeightPerGroupFloatHQO}},},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPoint},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPoint},
                'per_group': {
                    'sym': MXInt8Weight, 'asym': ShiftedMXUInt8Weight}},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPointMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPointMSE},
                'per_group': {
                    'sym': MXInt8WeightMSE, 'asym': ShiftedMXUInt8WeightMSE}}}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloat},
                'per_group': {
                    'sym': Fp8e4m3WeightSymmetricGroupQuant}}}},
    'float_ocp': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3OCPWeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloat},
                'per_group': {
                    'sym': Fp8e4m3OCPWeightSymmetricGroupQuant}},
            'mse': {
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloatMSE}}},
        'po2_scale': {
            'stats': {
                'per_group': {
                    'sym': MXFloat8e4m3Weight}},
            'mse': {
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFixedPointMSE},
                'per_group': {
                    'sym': MXFloat8e4m3WeightMSE}}}},
    'float_fnuz': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3FNUZWeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3FNUZWeightPerChannelFloat}}}}}

INPUT_QUANT_MAP = {
    'int': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat}},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloatMSE,
                        'asym': ShiftedUint8ActPerTensorFloatMSE}}},
            'po2_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPoint}},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPointMSE}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8DynamicActPerTensorFloat,
                        'asym': ShiftedUint8DynamicActPerTensorFloat},
                    'per_row': {
                        'sym': Int8DynamicActPerRowFloat,
                        'asym': ShiftedUint8DynamicActPerRowFloat},
                    'per_group': {
                        'sym': Int8DynamicActPerGroupFloat,
                        'asym': ShiftedUint8DynamicActPerGroupFloat}}},
            'po2_scale': {
                'stats': {
                    'per_row': {
                        'sym': Int8DynamicActPerRowFixedPoint,},
                    'per_group': {
                        'sym': MXInt8Act}}}}},
    'float': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3ActPerTensorFloat}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_group': {
                        'sym': Fp8e4m3DynamicActPerGroupFloat}}}},
        'no_scale': {
            'sym': Fp8e4m3Act,}},
    'float_ocp': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3OCPActPerTensorFloat}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_row': {
                        'sym': FP8e4m3OCPDynamicActPerRowFloat},
                    'per_group': {
                        'sym': Fp8e4m3OCPDynamicActPerGroupFloat}}},
            'po2_scale': {
                'stats': {
                    'per_row': {
                        'sym': FP8e4m3OCPDynamicActPerRowFixedPoint},
                    'per_group': {
                        'sym': MXFloat8e4m3Act}}}}},
    'float_fnuz': {
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3FNUZDynamicActPerTensorFloat},
                    'per_row': {
                        'sym': FP8e4m3FNUZDynamicActPerRowFloat}}}},
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3FNUZActPerTensorFloat}}}}}}


def generate_quantizers(
        dtype,
        weight_bit_width,
        weight_param_method,
        weight_scale_precision,
        weight_quant_type,
        weight_quant_granularity,
        weight_group_size,
        quantize_weight_zero_point,
        weight_quant_format='int',
        weight_group_dim=None,
        weight_scaling_impl_type='parameter_from_stats',
        input_bit_width=None,
        input_quant_format='',
        input_scale_precision=None,
        input_scale_type=None,
        input_param_method=None,
        input_quant_type=None,
        input_quant_granularity=None,
        input_group_size=None,
        kv_quant_type=None,
        kv_quant_granularity=None,
        quantize_input_zero_point=False,
        scale_rounding_func_type=None,
        device=None,
        weight_kwargs=None,
        input_kwargs=None,
        quant_attn_mode='mha',
        scaling_min_val=1e-4):
    """
    Replace float layers with quant layers in the target model
    """
    # Retrive base input and weight quantizers
    # match against custom float format
    if re.compile(r'e[1-8]m[1-8]').findall(weight_quant_format):
        format = re.compile(r'e[1-8]m[1-8]').findall(weight_quant_format)[0]
        weight_quant_format = weight_quant_format.replace('_' + format, '')
        weight_float_format = {
            'exponent_bit_width': int(format[1]), 'mantissa_bit_width': int(format[3])}
    else:
        weight_float_format = {}
    if re.compile(r'e[1-8]m[1-8]').findall(input_quant_format):
        format = re.compile(r'e[1-8]m[1-8]').findall(input_quant_format)[0]
        input_quant_format = input_quant_format.replace('_' + format, '')
        input_float_format = {
            'exponent_bit_width': int(format[1]), 'mantissa_bit_width': int(format[3])}
    else:
        input_float_format = {}

    weight_quant = WEIGHT_QUANT_MAP[weight_quant_format][weight_scale_precision][
        weight_param_method][weight_quant_granularity][weight_quant_type]

    if input_kwargs is None:
        input_kwargs = dict()

    if scale_rounding_func_type is not None:
        scale_rounding_func_dict = {'ceil': CeilSte, 'floor': FloorSte, 'round': RoundSte}
        scale_type = scale_rounding_func_dict[scale_rounding_func_type]
        input_kwargs = {**input_kwargs, **{'restrict_value_float_to_int_impl': scale_type}}

    if scaling_min_val is not None:
        input_kwargs = {**input_kwargs, **{'scaling_min_val': scaling_min_val}}

    if input_bit_width is not None and input_scale_type == 'no_scale':
        input_quant = sym_input_quant = linear_input_quant = INPUT_QUANT_MAP[input_quant_format][
            input_scale_type][input_quant_type]
    elif input_bit_width is not None:
        input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][input_scale_precision][
            input_param_method][input_quant_granularity][input_quant_type]
        # Some activations in MHA should always be symmetric
        sym_input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][
            input_scale_precision][input_param_method][input_quant_granularity]['sym']
        linear_input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][
            input_scale_precision][input_param_method][input_quant_granularity][input_quant_type]

        if kv_quant_type is not None:
            q_scaled_quant = attn_output_weights_quant = None

        else:
            q_scaled_quant = attn_output_weights_quant = sym_input_quant

        kv_quant_type = kv_quant_type if kv_quant_type is not None else input_quant_type
        kv_quant_granularity = kv_quant_granularity if kv_quant_granularity is not None else input_quant_granularity

        v_quant = k_transposed_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][
            input_scale_precision][input_param_method][kv_quant_granularity][kv_quant_type]

        extra_kwargs = {
            'bit_width': input_bit_width,
            'quantize_zero_point': quantize_input_zero_point,
            'dtype': dtype,
            'device': device}
        input_kwargs = {**input_kwargs, **extra_kwargs, **input_float_format}

        input_quant = input_quant.let(**input_kwargs)
        sym_input_quant = sym_input_quant.let(**input_kwargs)
        linear_input_quant = linear_input_quant.let(**input_kwargs)
        v_quant = v_quant.let(**input_kwargs)
        k_transposed_quant = k_transposed_quant.let(**input_kwargs)
        q_scaled_quant = q_scaled_quant.let(**input_kwargs) if q_scaled_quant is not None else None
        attn_output_weights_quant = attn_output_weights_quant.let(
            **input_kwargs) if attn_output_weights_quant is not None else None

    else:
        input_quant = None
        sym_input_quant = None
        linear_input_quant = None
        q_scaled_quant = attn_output_weights_quant = v_quant = k_transposed_quant = None

    # Modify the weight quantizer based on the arguments passed in
    weight_quant = weight_quant.let(
        **{
            'bit_width': weight_bit_width,
            'narrow_range': False,
            'quantize_zero_point': quantize_weight_zero_point},
        **weight_float_format)

    if scale_rounding_func_type is not None:
        scale_rounding_func_dict = {'ceil': CeilSte, 'floor': FloorSte, 'round': RoundSte}
        scale_type = scale_rounding_func_dict[scale_rounding_func_type]
        weight_quant = weight_quant.let(**{'restrict_value_float_to_int_impl': scale_type})

    if weight_group_dim is not None:
        weight_quant = weight_quant.let(**{'group_dim': weight_group_dim})

    if scaling_min_val is not None:
        weight_quant = weight_quant.let(**{'scaling_min_val': scaling_min_val})

    if weight_kwargs is not None:
        weight_quant = weight_quant.let(**weight_kwargs)

    # Set the group_size is we're doing groupwise quantization
    if weight_quant_granularity == 'per_group':
        weight_quant = weight_quant.let(**{'group_size': weight_group_size})
    # weight scale is converted to a standalone parameter

    weight_quant = weight_quant.let(scaling_impl_type=weight_scaling_impl_type)
    # weight zero-point is converted to a standalone parameter
    # This is done already by default in the per_group quantizer
    if weight_quant_type == 'asym' and weight_scaling_impl_type == 'parameter_from_stats':
        weight_quant = weight_quant.let(zero_point_impl=ParameterFromStatsFromParameterZeroPoint)

    if quant_attn_mode == 'sdpa':
        kv_permute_dims = (0, 1, 3, 2)
        kv_broadcastable_shape_lambda = lambda x, shape: x.view(shape[0], shape[1], 1, shape[-1])
    elif quant_attn_mode == 'mha':
        kv_permute_dims = (0, 2, 1)
        kv_broadcastable_shape_lambda = lambda x, shape: x.view(shape[0], 1, shape[-1])

    # Modify the input quantizers based on the arguments passed in
    if input_bit_width is not None:
        # Input Quant
        if input_quant_granularity == 'per_row':
            input_quant = input_quant.let(
                **{
                    'dynamic_scaling_broadcastable_fn': lambda x,
                                                        shape: x.view(*shape[:-1], 1),
                    'permute_dims': None,
                    'stats_reduce_dim': 1})
        elif input_quant_granularity == 'per_group':
            input_quant = input_quant.let(**{'group_size': input_group_size})

        # QKV/Softmax Quant
        if kv_quant_granularity == 'per_row':
            q_scaled_quant = q_scaled_quant.let(
                **{
                    'dynamic_scaling_broadcastable_fn': lambda x,
                                                        shape: x.view(*shape[:-1], 1),
                    'permute_dims': None,
                    'stats_reduce_dim': 1}) if q_scaled_quant is not None else None
            v_quant = v_quant.let(
                **{
                    'dynamic_scaling_broadcastable_fn': kv_broadcastable_shape_lambda,
                    'permute_dims': kv_permute_dims,
                    'stats_reduce_dim': 1})
            k_transposed_quant = k_transposed_quant.let(
                **{
                    'dynamic_scaling_broadcastable_fn': kv_broadcastable_shape_lambda,
                    'permute_dims': kv_permute_dims,
                    'stats_reduce_dim': 1})
        elif kv_quant_granularity == 'per_group':
            q_scaled_quant = q_scaled_quant.let(
                **{
                    'group_dim': -1, 'group_size': input_group_size
                }) if q_scaled_quant is not None else None
            v_quant = v_quant.let(**{'group_dim': -1, 'group_size': input_group_size})
            k_transposed_quant = k_transposed_quant.let(
                **{
                    'group_dim': -2, 'group_size': input_group_size})
        v_quant = k_transposed_quant
        attn_output_weights_quant = q_scaled_quant

        # Input to Linear Layer Quant
        if input_quant_granularity == 'per_row':
            linear_input_quant = linear_input_quant.let(
                **{
                    'dynamic_scaling_broadcastable_fn': lambda x,
                                                        shape: x.view(*shape[:-1], 1),
                    'permute_dims': None,
                    'stats_reduce_dim': 1})
        elif input_quant_granularity == 'per_group':
            linear_input_quant = linear_input_quant.let(
                **{
                    'group_dim': -1, 'group_size': input_group_size})
    return linear_input_quant, weight_quant, input_quant, q_scaled_quant, k_transposed_quant, v_quant, attn_output_weights_quant


def generate_quant_maps(
        linear_input_quant,
        weight_quant,
        input_quant,
        q_scaled_quant,
        k_transposed_quant,
        v_quant,
        attn_output_weights_quant,
        dtype,
        device,
        input_quant_format,
        quantize_embedding):

    quant_linear_kwargs = {
        'input_quant': linear_input_quant,
        'weight_quant': weight_quant,
        'dtype': dtype,
        'device': device}
    quant_conv_kwargs = {
        'input_quant': input_quant, 'weight_quant': weight_quant, 'dtype': dtype, 'device': device}

    quant_mha_kwargs = {
        'in_proj_input_quant': input_quant,
        'in_proj_weight_quant': weight_quant,
        'in_proj_bias_quant': None,
        'softmax_input_quant': None,
        'attn_output_weights_quant': attn_output_weights_quant,
        'attn_output_weights_signed': 'float' in input_quant_format,
        'q_scaled_quant': q_scaled_quant,
        'k_transposed_quant': k_transposed_quant,
        'v_quant': v_quant,
        'out_proj_input_quant': input_quant,
        'out_proj_weight_quant': weight_quant,
        'out_proj_bias_quant': None,
        'out_proj_output_quant': None,
        'batch_first': True,
        # activation equalization requires packed_in_proj
        # since it supports only self-attention
        'packed_in_proj': True,
        'dtype': dtype,
        'device': device}

    quant_sdpa_kwargs = {
        'softmax_input_quant': None,
        'attn_output_weights_quant': attn_output_weights_quant,
        'attn_output_weights_signed': 'float' in input_quant_format,
        'q_scaled_quant': q_scaled_quant,
        'k_transposed_quant': k_transposed_quant,
        'v_quant': v_quant,
        'attn_output_quant': None,
        'dtype': dtype,
        'device': device}

    layer_map = {
        nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
        nn.Conv2d: (qnn.QuantConv2d, quant_conv_kwargs),
        'diffusers.models.lora.LoRACompatibleLinear':
            (LoRACompatibleQuantLinear, quant_linear_kwargs),
        'diffusers.models.lora.LoRACompatibleConv': (LoRACompatibleQuantConv2d, quant_conv_kwargs),
        nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs),
        qnn.ScaledDotProductAttention: (qnn.QuantScaledDotProductAttention, quant_sdpa_kwargs)}

    if quantize_embedding:
        quant_embedding_kwargs = {'weight_quant': weight_quant, 'dtype': dtype, 'device': device}
        layer_map[nn.Embedding] = (qnn.QuantEmbedding, quant_embedding_kwargs)
    return layer_map


def quantize_model(
        model,
        dtype,
        weight_bit_width,
        weight_param_method,
        weight_scale_precision,
        weight_quant_type,
        weight_quant_granularity,
        weight_group_size,
        quantize_weight_zero_point,
        weight_quant_format='int',
        name_blacklist=None,
        input_bit_width=None,
        input_quant_format='',
        input_scale_precision=None,
        input_scale_type=None,
        input_param_method=None,
        input_quant_type=None,
        input_quant_granularity=None,
        input_group_size=None,
        quantize_input_zero_point=False,
        quantize_embedding=False,
        device=None,
        weight_kwargs=None,
        input_kwargs=None):

    linear_input_quant, weight_quant, input_quant, q_scaled_quant, k_transposed_quant, v_quant, attn_output_weights_quant = generate_quantizers(
        dtype=dtype,
        weight_bit_width=weight_bit_width,
        weight_param_method=weight_param_method,
        weight_scale_precision=weight_scale_precision,
        weight_quant_type=weight_quant_type,
        weight_quant_granularity=weight_quant_granularity,
        weight_group_size=weight_group_size,
        quantize_weight_zero_point=quantize_weight_zero_point,
        weight_quant_format=weight_quant_format,
        input_bit_width=input_bit_width,
        input_quant_format=input_quant_format,
        input_scale_precision=input_scale_precision,
        input_scale_type=input_scale_type,
        input_param_method=input_param_method,
        input_quant_type=input_quant_type,
        input_quant_granularity=input_quant_granularity,
        input_group_size=input_group_size,
        quantize_input_zero_point=quantize_input_zero_point,
        device=device,
        weight_kwargs=weight_kwargs,
        input_kwargs=input_kwargs)
    layer_map = generate_quant_maps(
        linear_input_quant,
        weight_quant,
        input_quant,
        q_scaled_quant,
        k_transposed_quant,
        v_quant,
        attn_output_weights_quant,
        dtype,
        device,
        input_quant_format,
        quantize_embedding)
    model = layerwise_quantize(
        model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
    return model
