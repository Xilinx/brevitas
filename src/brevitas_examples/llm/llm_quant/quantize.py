"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from torch import nn
from transformers.models.opt.modeling_opt import OPTAttention

from brevitas import nn as qnn
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.quantize import layerwise_quantize
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
from brevitas_examples.llm.llm_quant.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.llm.llm_quant.quantizers import ShiftedUintWeightAsymmetricGroupQuant

WEIGHT_QUANT_MAP = {
    'float32': {
        'stats': {
            'per_tensor': {
                'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
            'per_channel': {
                'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat},
            'per_group': {
                'sym': IntWeightSymmetricGroupQuant, 'asym': ShiftedUintWeightAsymmetricGroupQuant},
        },
        'mse': {
            'per_tensor': {
                'sym': Int8WeightPerTensorFloatMSE, 'asym': ShiftedUint8WeightPerTensorFloatMSE},
            'per_channel': {
                'sym': Int8WeightPerChannelFloatMSE, 'asym': ShiftedUint8WeightPerChannelFloatMSE},
        },},
    'po2': {
        'stats': {
            'per_tensor': {
                'sym': Int8WeightPerTensorFixedPoint},
            'per_channel': {
                'sym': Int8WeightPerChannelFixedPoint},},
        'mse': {
            'per_tensor': {
                'sym': Int8WeightPerTensorFixedPointMSE},
            'per_channel': {
                'sym': Int8WeightPerChannelFixedPointMSE},},}}

INPUT_QUANT_MAP = {
    'float32': {
        'stats': {
            'per_tensor': {
                'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat},},
        'mse': {
            'per_tensor': {
                'sym': Int8ActPerTensorFloatMSE, 'asym': ShiftedUint8ActPerTensorFloatMSE},},},
    'po2': {
        'stats': {
            'per_tensor': {
                'sym': Int8ActPerTensorFixedPoint},},
        'mse': {
            'per_tensor': {
                'sym': Int8ActPerTensorFixedPointMSE},},}}


def quantize_model(
        model,
        weight_bit_width,
        weight_param_method,
        weight_scale_type,
        weight_quant_type,
        weight_quant_granularity,
        weight_group_size,
        quantize_weight_zero_point,
        input_bit_width=None,
        input_scale_type=None,
        input_param_method=None,
        input_quant_type=None,
        input_quant_granularity=None,
        quantize_input_zero_point=False):
    """
    Replace float layers with quant layers in the target model
    """
    # Retrive base input and weight quantizers
    weight_quant = WEIGHT_QUANT_MAP[weight_scale_type][weight_param_method][
        weight_quant_granularity][weight_quant_type]
    if input_bit_width is not None:
        input_quant = INPUT_QUANT_MAP[input_scale_type][input_param_method][
            input_quant_granularity][input_quant_type]
        # Some activations in MHA should always be symmetric
        sym_input_quant = INPUT_QUANT_MAP[input_scale_type][input_param_method][
            input_quant_granularity]['sym']
    else:
        input_quant = None
        sym_input_quant = None

    # Modify the weight quantizer based on the arguments passed in
    weight_quant = weight_quant.let(
        **{
            'bit_width': weight_bit_width,
            'block_size': weight_group_size,
            'quantize_zero_point': quantize_weight_zero_point})
    # weight scale is converted to a standalone parameter
    # This is done already by default in the per_group quantizer
    if weight_quant_granularity != 'per_group':
        weight_quant = weight_quant.let(weight_scale_impl_type='parameter_from_stats')
    # weight zero-point is converted to a standalone parameter
    # This is done already by default in the per_group quantizer
    if weight_quant_type == 'asym' and weight_quant_granularity != 'per_group':
        weight_quant = weight_quant.let(zero_point_impl=ParameterFromStatsFromParameterZeroPoint)

    # Modify the input quantizers based on the arguments passed in
    if input_quant is not None:
        input_quant = input_quant.let(
            **{
                'bit_width': input_bit_width, 'quantize_zero_point': quantize_input_zero_point})
    if sym_input_quant is not None:
        sym_input_quant = sym_input_quant.let(
            **{
                'bit_width': input_bit_width, 'quantize_zero_point': quantize_input_zero_point})

    quant_linear_kwargs = {'input_quant': input_quant, 'weight_quant': weight_quant}

    quant_mha_kwargs = {
        'in_proj_input_quant': input_quant,
        'in_proj_weight_quant': weight_quant,
        'in_proj_bias_quant': None,
        'softmax_input_quant': None,
        'attn_output_weights_quant': sym_input_quant,
        'attn_output_weights_signed': False,
        'q_scaled_quant': sym_input_quant,
        'k_transposed_quant': sym_input_quant,
        'v_quant': sym_input_quant,
        'out_proj_input_quant': input_quant,
        'out_proj_weight_quant': weight_quant,
        'out_proj_bias_quant': None,
        'out_proj_output_quant': None,
        'batch_first': True,
        'packed_in_proj': False}

    layer_map = {
        nn.modules.linear.NonDynamicallyQuantizableLinear: (qnn.QuantLinear, quant_linear_kwargs),
        nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
        nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs)}
    layerwise_quantize(model=model, compute_layer_map=layer_map)
