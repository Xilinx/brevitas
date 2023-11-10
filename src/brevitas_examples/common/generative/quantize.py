"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
import re

from torch import nn

from brevitas import nn as qnn
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant.experimental.float import Fp8e4m3Act
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPointMSE
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.common.generative.nn import LoRACompatibleQuantConv2d
from brevitas_examples.common.generative.nn import LoRACompatibleQuantLinear
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Int8ActDynamicPerGroupFloat
from brevitas_examples.common.generative.quantizers import Int8ActDynamicPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8ActDynamicPerTensorFloat
from brevitas_examples.common.generative.quantizers import Int8ActPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8ActPerRowFloatMSE
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import ShiftedUint8ActPerRowFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8ActPerRowFloatMSE
from brevitas_examples.common.generative.quantizers import ShiftedUintWeightAsymmetricGroupQuant

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
                    'asym': ShiftedUintWeightAsymmetricGroupQuant},},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatMSE,
                    'asym': ShiftedUint8WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatMSE,
                    'asym': ShiftedUint8WeightPerChannelFloatMSE},},},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPoint},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPoint},},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPointMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPointMSE},},}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloat},
                'per_group': {
                    'sym': Fp8e4m3WeightSymmetricGroupQuant}},}}}

INPUT_QUANT_MAP = {
    'int': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat},
                    'per_row': {
                        'sym': Int8ActPerRowFloat, 'asym': ShiftedUint8ActPerRowFloat},},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloatMSE, 'asym': ShiftedUint8ActPerTensorFloatMSE},
                    'per_row': {
                        'sym': Int8ActPerRowFloatMSE, 'asym': ShiftedUint8ActPerRowFloatMSE},},},
            'po2_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPoint},},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPointMSE},},}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActDynamicPerTensorFloat},
                    'per_row': {
                        'sym': Int8ActDynamicPerRowFloat},
                    'per_group': {
                        'sym': Int8ActDynamicPerGroupFloat},}}}},
    'float': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3ActPerTensorFloat},}}},
        'no_scale': {
            'sym': Fp8e4m3Act,}}}


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
        seqlen=None):
    """
    Replace float layers with quant layers in the target model
    """
    # Retrive base input and weight quantizers

    # match against custom float format
    if re.compile(r'e[1-8]m[1-8]').match(weight_quant_format):
        weight_float_format = {
            'exponent_bit_width': int(weight_quant_format[1]),
            'mantissa_bit_width': int(weight_quant_format[3])}
        weight_quant_format = 'float'
    else:
        weight_float_format = {}
    if re.compile(r'e[1-8]m[1-8]').match(input_quant_format):
        input_float_format = {
            'exponent_bit_width': int(input_quant_format[1]),
            'mantissa_bit_width': int(input_quant_format[3])}
        input_quant_format = 'float'
    else:
        input_float_format = {}

    weight_quant = WEIGHT_QUANT_MAP[weight_quant_format][weight_scale_precision][
        weight_param_method][weight_quant_granularity][weight_quant_type]
    if input_bit_width is not None and input_scale_type == 'no_scale':
        input_quant = sym_input_quant = linear_2d_input_quant = INPUT_QUANT_MAP[input_quant_format][
            input_scale_type][input_quant_type]
    elif input_bit_width is not None:
        input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][input_scale_precision][
            input_param_method][input_quant_granularity][input_quant_type]
        # Some activations in MHA should always be symmetric
        sym_input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][
            input_scale_precision][input_param_method][input_quant_granularity]['sym']
        # Linear layers with 2d input should always be per tensor or per group, as there is no row dimension
        if input_quant_granularity == 'per_tensor' or input_quant_granularity == 'per_row':
            linear_2d_input_quant = INPUT_QUANT_MAP[input_quant_format][input_scale_type][
                input_scale_precision][input_param_method]['per_tensor'][input_quant_type]
        else:
            assert input_quant_granularity == 'per_group'
            linear_2d_input_quant = input_quant
    else:
        input_quant = None
        sym_input_quant = None
        linear_2d_input_quant = None

    # Modify the weight quantizer based on the arguments passed in
    weight_quant = weight_quant.let(
        **{
            'narrow_range': False,
            'block_size': weight_group_size,
            'quantize_zero_point': quantize_weight_zero_point},
        **weight_float_format)
    # weight scale is converted to a standalone parameter
    # This is done already by default in the per_group quantizer
    if weight_quant_granularity != 'per_group':
        weight_quant = weight_quant.let(scaling_impl_type='parameter_from_stats')
    # weight zero-point is converted to a standalone parameter
    # This is done already by default in the per_group quantizer
    if weight_quant_type == 'asym' and weight_quant_granularity != 'per_group':
        weight_quant = weight_quant.let(zero_point_impl=ParameterFromStatsFromParameterZeroPoint)

    # Modify the input quantizers based on the arguments passed in
    if input_quant is not None:
        input_quant = input_quant.let(
            **{
                'bit_width': input_bit_width,
                'quantize_zero_point': quantize_input_zero_point,
                'dtype': dtype,},
            **input_float_format)
        if input_scale_type == 'static' and input_quant_granularity == 'per_row':
            # QuantMHA internally always uses Seq, B, E
            input_quant = input_quant.let(
                **{
                    'per_channel_broadcastable_shape': (seqlen, 1, 1),
                    'scaling_stats_permute_dims': (0, 1, 2)})
        elif input_scale_type == 'dynamic':
            if input_quant_granularity == 'per_tensor':
                input_quant = input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (1, -1, 1),
                        'permute_dims': (1, 0, 2),
                        'stats_reduce_dim': 1})
            elif input_quant_granularity == 'per_row':
                input_quant = input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (seqlen, -1, 1),
                        'permute_dims': (1, 0, 2),
                        'stats_reduce_dim': 2})
            elif input_quant_granularity == 'per_group':
                input_quant = input_quant.let(**{'group_dim': 2, 'group_size': input_group_size})
    if sym_input_quant is not None:
        sym_input_quant = sym_input_quant.let(
            **{
                'bit_width': input_bit_width,
                'quantize_zero_point': quantize_input_zero_point,
                'dtype': dtype},
            **input_float_format)
        if input_scale_type == 'static' and input_quant_granularity == 'per_row':
            q_scaled_quant = sym_input_quant.let(
                **{
                    'per_channel_broadcastable_shape': (1, seqlen, 1),
                    'scaling_stats_permute_dims': (1, 0, 2)})
            k_transposed_quant = sym_input_quant.let(
                **{
                    'per_channel_broadcastable_shape': (1, 1, seqlen),
                    'scaling_stats_permute_dims': (2, 0, 1)})
            v_quant = q_scaled_quant
            attn_output_weights_quant = q_scaled_quant
        elif input_scale_type == 'dynamic':
            if input_quant_granularity == 'per_tensor':
                q_scaled_quant = sym_input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (-1, 1, 1),
                        'permute_dims': None,
                        'stats_reduce_dim': 1})
                k_transposed_quant = sym_input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (-1, 1, 1),
                        'permute_dims': None,
                        'stats_reduce_dim': 1})
            elif input_quant_granularity == 'per_row':
                q_scaled_quant = sym_input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (-1, seqlen, 1),
                        'permute_dims': None,
                        'stats_reduce_dim': 2})
                k_transposed_quant = sym_input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (-1, 1, seqlen),
                        'permute_dims': None,
                        'stats_reduce_dim': 1})
            elif input_quant_granularity == 'per_group':
                q_scaled_quant = sym_input_quant.let(
                    **{
                        'group_dim': 2, 'group_size': input_group_size})
                k_transposed_quant = sym_input_quant.let(
                    **{
                        'group_dim': 1, 'group_size': input_group_size})
            v_quant = q_scaled_quant
            attn_output_weights_quant = q_scaled_quant
        else:
            q_scaled_quant = v_quant = k_transposed_quant = attn_output_weights_quant = sym_input_quant
    else:
        q_scaled_quant = v_quant = k_transposed_quant = attn_output_weights_quant = None
    if linear_2d_input_quant is not None:
        linear_2d_input_quant = linear_2d_input_quant.let(
            **{
                'bit_width': input_bit_width,
                'quantize_zero_point': quantize_input_zero_point,
                'dtype': dtype},
            **input_float_format)
        if input_scale_type == 'dynamic':
            # Note: this breaks if applied to 3d Linear inputs,
            # in case standard MHA layers haven't been inserted
            if input_quant_granularity == 'per_tensor' or input_quant_granularity == 'per_row':
                linear_2d_input_quant = linear_2d_input_quant.let(
                    **{
                        'dynamic_scaling_broadcastable_shape': (-1, 1),
                        'permute_dims': None,
                        'stats_reduce_dim': 1})
            elif input_quant_granularity == 'per_group':
                linear_2d_input_quant = linear_2d_input_quant.let(
                    **{
                        'group_dim': 1, 'group_size': input_group_size})

    quant_linear_kwargs = {
        'input_quant': linear_2d_input_quant,
        'weight_quant': weight_quant,
        'weight_bit_width': weight_bit_width,
        'dtype': dtype}
    quant_conv_kwargs = {
        'input_quant': input_quant,
        'weight_quant': weight_quant,
        'weight_bit_width': weight_bit_width,
        'dtype': dtype}

    quant_mha_kwargs = {
        'in_proj_input_quant': input_quant,
        'in_proj_weight_quant': weight_quant,
        'in_proj_bias_quant': None,
        'softmax_input_quant': None,
        'attn_output_weights_quant': attn_output_weights_quant,
        'attn_output_weights_signed': input_quant_format == 'float',
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
        'dtype': dtype}

    layer_map = {
        nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
        nn.Conv2d: (qnn.QuantConv2d, quant_conv_kwargs),
        'diffusers.models.lora.LoRACompatibleLinear':
            (LoRACompatibleQuantLinear, quant_linear_kwargs),
        'diffusers.models.lora.LoRACompatibleConv': (LoRACompatibleQuantConv2d, quant_conv_kwargs),
        nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs)}

    if quantize_embedding:
        quant_embedding_kwargs = {'weight_quant': weight_quant, 'dtype': dtype}
        layer_map[nn.Embedding] = (qnn.QuantEmbedding, quant_embedding_kwargs)

    layerwise_quantize(model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
