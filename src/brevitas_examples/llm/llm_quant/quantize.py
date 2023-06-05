"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from torch import nn

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
    weight_quant = WEIGHT_QUANT_MAP[weight_scale_type][weight_param_method][
        weight_quant_granularity][weight_quant_type]
    if input_bit_width is not None:
        input_quant = INPUT_QUANT_MAP[input_scale_type][input_param_method][
            input_quant_granularity][input_quant_type]
    else:
        input_quant = None

    quant_linear_kwargs = {
        'input_quant': input_quant,
        'input_bit_width': input_bit_width,
        'weight_quant': weight_quant,
        'weight_bit_width': weight_bit_width,
        # weight scale is always converted to a standalone parameter when possible
        'weight_scale_impl_type': 'parameter_from_stats',  # ignored args if unused
        'weight_block_size': weight_group_size,
        'weight_quantize_zero_point': quantize_weight_zero_point,
        'input_quantize_zero_point': quantize_input_zero_point}
    # weight zero-point is always converted to a standalone parameter
    # This done by default already in the group quantizer
    if weight_quant_type == 'asym' and weight_quant_granularity != 'per_group':
        quant_linear_kwargs['zero_point_impl'] = ParameterFromStatsFromParameterZeroPoint
    layer_map = {nn.Linear: (qnn.QuantLinear, quant_linear_kwargs)}
    layerwise_quantize(model=model, compute_layer_map=layer_map)
