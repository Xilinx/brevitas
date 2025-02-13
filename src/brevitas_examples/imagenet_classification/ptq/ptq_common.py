# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import math

import torch
from tqdm import tqdm

from brevitas.core.function_wrapper.shape import OverBatchOverTensorView
from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import norm_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import quantize_flexml
from brevitas.inject import value
import brevitas.nn as qnn
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloatMSE
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloatMSE
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloatMSE
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloatMSE
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
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.common.axe import A2GPFQ
from brevitas_examples.common.axe import A2GPTQ
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat


# Every element of the Batch will have its own scale factor and zero point
class CNNShiftedUint8DynamicActPerTensorFloat(ShiftedUint8DynamicActPerTensorFloat):
    scaling_stats_input_view_shape_impl = OverBatchOverTensorView
    scaling_stats_permute_dims = None
    stats_reduce_dim = 1
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(shape[0], *[1 for _ in range(len(shape[1:]))])


class CNNInt8DynamicActPerTensorFloat(Int8DynamicActPerTensorFloat):
    scaling_stats_input_view_shape_impl = OverBatchOverTensorView
    scaling_stats_permute_dims = None
    stats_reduce_dim = 1
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(shape[0], *[1 for _ in range(len(shape[1:]))])


QUANTIZE_MAP = {'layerwise': layerwise_quantize, 'fx': quantize, 'flexml': quantize_flexml}

BIAS_BIT_WIDTH_MAP = {32: Int32Bias, 16: Int16Bias, None: None}

WEIGHT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat}},
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
                    'asym': ShiftedUint8WeightPerChannelFloatHQO}}},
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
                    'sym': MXInt8WeightMSE, 'asym': ShiftedMXUInt8WeightMSE}},}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloatMSE}}}},
    'float_ocp': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3OCPWeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Fp8e4m3OCPWeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloatMSE}}},
        'po2_scale': {
            'stats': {
                'per_group': {
                    'sym': MXFloat8e4m3Weight}},
            'mse': {
                'per_group': {
                    'sym': MXFloat8e4m3WeightMSE}}}}}

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
                        'sym': Int8ActPerTensorFixedPoint,
                        'asym': ShiftedUint8ActPerTensorFixedPoint},},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPointMSE}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': CNNInt8DynamicActPerTensorFloat,
                        'asym': CNNShiftedUint8DynamicActPerTensorFloat}}},
            'po2_scale': {
                'stats': {
                    'per_group': {
                        'sym': MXInt8Act}}}}},
    'float': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3ActPerTensorFloat}},
                'mse': {
                    'per_tensor': {
                        'sym': Fp8e4m3ActPerTensorFloatMSE}}}}},
    'float_ocp': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3OCPActPerTensorFloat}},
                'mse': {
                    'per_tensor': {
                        'sym': Fp8e4m3OCPActPerTensorFloatMSE}}}},
        'dynamic': {
            'po2_scale': {
                'stats': {
                    'per_group': {
                        'sym': MXFloat8e4m3Act}}}}}}


def quantize_model(
        model,
        backend,
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        weight_quant_granularity,
        act_quant_percentile,
        act_quant_type,
        scale_factor_type,
        quant_format,
        layerwise_first_last_bit_width=8,
        layerwise_first_last_mantissa_bit_width=4,
        layerwise_first_last_exponent_bit_width=3,
        weight_mantissa_bit_width=4,
        weight_exponent_bit_width=3,
        act_mantissa_bit_width=4,
        act_exponent_bit_width=3,
        weight_narrow_range=False,
        weight_param_method='stats',
        act_param_method='stats',
        weight_quant_type='sym',
        act_quant_granularity='per_tensor',
        act_scale_computation_type='static',
        uint_sym_act_for_unsigned_values=True,
        dtype=torch.float32,
        device='cpu'):
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]
    weight_scale_type = scale_factor_type
    act_scale_type = scale_factor_type

    # We check all of the provided values are positive integers
    check_positive_int(
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        layerwise_first_last_bit_width,
        layerwise_first_last_mantissa_bit_width,
        layerwise_first_last_exponent_bit_width,
        weight_mantissa_bit_width,
        weight_exponent_bit_width,
        act_mantissa_bit_width,
        act_exponent_bit_width)

    if act_scale_computation_type == 'dynamic' and backend != 'layerwise':
        assert bias_bit_width is None, "Bias quantization is not supported with dynamic activation quantization"

    weight_quant_format = quant_format
    act_quant_format = quant_format

    def layerwise_bit_width_fn(module, base_bit_width, first_last_bit_width):
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            return first_last_bit_width
        elif isinstance(module, torch.nn.Linear) and module.out_features == 1000:
            return first_last_bit_width
        else:
            return base_bit_width

    @value
    def layerwise_bit_width_fn_act_exponent(module):
        return layerwise_bit_width_fn(
            module, act_exponent_bit_width, layerwise_first_last_exponent_bit_width)

    @value
    def layerwise_bit_width_fn_act_mantissa(module):
        return layerwise_bit_width_fn(
            module, act_mantissa_bit_width, layerwise_first_last_mantissa_bit_width)

    @value
    def layerwise_bit_width_fn_weight_exponent(module):
        return layerwise_bit_width_fn(
            module, weight_exponent_bit_width, layerwise_first_last_exponent_bit_width)

    @value
    def layerwise_bit_width_fn_weight_mantissa(module):
        return layerwise_bit_width_fn(
            module, weight_mantissa_bit_width, layerwise_first_last_mantissa_bit_width)

    @value
    def layerwise_bit_width_fn_act(module):
        return layerwise_bit_width_fn(module, act_bit_width, layerwise_first_last_bit_width)

    @value
    def layerwise_bit_width_fn_weight(module):
        return layerwise_bit_width_fn(module, weight_bit_width, layerwise_first_last_bit_width)

    # Missing fix for backend =! layerwise
    # Missing fix for name_shadowing for all variables
    weight_bit_width_dict = {}
    act_bit_width_dict = {}
    if quant_format == 'int' and backend == 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = layerwise_bit_width_fn_weight
        if act_bit_width is not None:
            act_bit_width_dict['act_bit_width'] = layerwise_bit_width_fn_act
        else:
            act_bit_width_dict['act_bit_width'] = None
    elif quant_format == 'int' and backend != 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = act_bit_width

    if 'float' in quant_format and backend == 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = layerwise_bit_width_fn_weight
        act_bit_width_dict['act_bit_width'] = layerwise_bit_width_fn_act
        weight_bit_width_dict['weight_mantissa_bit_width'] = layerwise_bit_width_fn_weight_mantissa
        weight_bit_width_dict['weight_exponent_bit_width'] = layerwise_bit_width_fn_weight_exponent
        act_bit_width_dict['act_mantissa_bit_width'] = layerwise_bit_width_fn_act_mantissa
        act_bit_width_dict['act_exponent_bit_width'] = layerwise_bit_width_fn_act_exponent
    elif 'float' in quant_format and backend != 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = act_bit_width
        weight_bit_width_dict['weight_mantissa_bit_width'] = weight_mantissa_bit_width
        weight_bit_width_dict['weight_exponent_bit_width'] = weight_exponent_bit_width
        act_bit_width_dict['act_mantissa_bit_width'] = act_mantissa_bit_width
        act_bit_width_dict['act_exponent_bit_width'] = act_exponent_bit_width


    quant_layer_map, quant_layerwise_layer_map, quant_act_map, quant_identity_map = create_quant_maps(dtype=dtype,
                            device=device,
                            uint_sym_act_for_unsigned_values=uint_sym_act_for_unsigned_values,
                            bias_bit_width=bias_bit_width,
                            weight_param_method=weight_param_method,
                            weight_scale_type=weight_scale_type,
                            weight_quant_type=weight_quant_type,
                            weight_quant_granularity=weight_quant_granularity,
                            weight_narrow_range=weight_narrow_range,
                            weight_quant_format=weight_quant_format,
                            act_quant_format=act_quant_format,
                            act_scale_type=act_scale_type,
                            act_param_method=act_param_method,
                            act_quant_type=act_quant_type,
                            act_quant_granularity=act_quant_granularity,
                            act_quant_percentile=act_quant_percentile,
                            act_scale_computation_type=act_scale_computation_type,
                            **weight_bit_width_dict,
                            **act_bit_width_dict)

    if backend != 'layerwise':
        # Fx and flexml backend requires three mappings for quantization
        quantize_kwargs = {
            'compute_layer_map': quant_layer_map,
            'quant_act_map': quant_act_map,
            'quant_identity_map': quant_identity_map}
    else:
        # Layerwise requires only the compute layer mapping
        quantize_kwargs = {'compute_layer_map': quant_layerwise_layer_map}

    quant_model = quantize_fn(model, **quantize_kwargs)
    return quant_model


def create_quant_maps(
        dtype,
        bias_bit_width,
        weight_bit_width,
        weight_param_method,
        weight_scale_type,
        weight_quant_type,
        weight_quant_granularity,
        weight_narrow_range,
        weight_quant_format,
        act_quant_format,
        uint_sym_act_for_unsigned_values=True,
        weight_mantissa_bit_width=None,
        weight_exponent_bit_width=None,
        act_mantissa_bit_width=None,
        act_exponent_bit_width=None,
        act_bit_width=None,
        act_scale_type=None,
        act_scale_computation_type=None,
        act_param_method=None,
        act_quant_type=None,
        act_quant_granularity=None,
        act_quant_percentile=None,
        device='cpu'):
    """
    Starting from pre-defined quantizers, modify them to match the desired configuration
    """

    def kwargs_prefix(prefix, weight_kwargs):
        return {prefix + k: v for k, v in weight_kwargs.items()}

    weight_bit_width_dict = {'bit_width': weight_bit_width}
    if 'float' in weight_quant_format:
        weight_bit_width_dict['exponent_bit_width'] = weight_exponent_bit_width
        weight_bit_width_dict['mantissa_bit_width'] = weight_mantissa_bit_width

    act_bit_width_dict = {'bit_width': act_bit_width}
    if 'float' in act_quant_format:
        act_bit_width_dict['exponent_bit_width'] = act_exponent_bit_width
        act_bit_width_dict['mantissa_bit_width'] = act_mantissa_bit_width

    # Retrieve base input, weight, and bias quantizers
    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width] if act_bit_width is not None else None
    weight_quant = WEIGHT_QUANT_MAP[weight_quant_format][weight_scale_type][weight_param_method][
        weight_quant_granularity][weight_quant_type]
    weight_quant = weight_quant.let(**weight_bit_width_dict)

    if act_bit_width is not None:
        act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_computation_type][act_scale_type][
            act_param_method][act_quant_granularity][act_quant_type]
        # Some activations in MHA should always be symmetric
        sym_act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_computation_type][
            act_scale_type][act_param_method][act_quant_granularity]['sym']

        act_quant = act_quant.let(**act_bit_width_dict)
        act_quant = act_quant.let(**{'dtype': dtype, 'device': device})
        sym_act_quant = sym_act_quant.let(**act_bit_width_dict)
        sym_act_quant = sym_act_quant.let(**{'dtype': dtype, 'device': device})
    else:
        act_quant = None
        sym_act_quant = None

    # Modify the weight quantizer based on the arguments passed in
    weight_quant = weight_quant.let(
        **{
            'narrow_range': weight_narrow_range,
            'scaling_impl': ParameterFromStatsFromParameterScaling})
    if weight_quant_type == 'asym':
        weight_quant = weight_quant.let(zero_point_impl=ParameterFromStatsFromParameterZeroPoint)
    if act_quant is not None:
        act_quant = act_quant.let(**{'high_percentile_q': act_quant_percentile})
        if act_quant_type == 'asym' and act_quant_percentile is not None:
            act_quant = act_quant.let(**{'low_percentile_q': 100 - act_quant_percentile})
    if sym_act_quant is not None:
        sym_act_quant = sym_act_quant.let(**{'high_percentile_q': act_quant_percentile})

    weight_quant_dict = {'weight_quant': weight_quant}

    quant_wbiol_kwargs = {
        **weight_quant_dict,
        'dtype': dtype,
        'device': device,
        'return_quant_tensor': False,
        'bias_quant': bias_quant}

    # yapf: disable
    quant_mha_kwargs = {
        **kwargs_prefix('in_proj_', weight_quant_dict),
        **kwargs_prefix('out_proj_', weight_quant_dict),
        'in_proj_input_quant': None,
        'in_proj_bias_quant': bias_quant,
        'softmax_input_quant': None,
        'attn_output_weights_quant': sym_act_quant,
        'q_scaled_quant': sym_act_quant,
        'k_transposed_quant': sym_act_quant,
        'v_quant': sym_act_quant,
        'out_proj_input_quant': act_quant,
        'out_proj_bias_quant': bias_quant,
        'out_proj_output_quant': None,
        # activation equalization requires packed_in_proj
        # since it supports only self-attention
        'packed_in_proj': True,
        'dtype': dtype,
        'device': device,
        'return_quant_tensor': False}
    # yapf: enable

    quant_act_kwargs = {'act_quant': act_quant, 'return_quant_tensor': True}
    # For potentially unsigned activations, we create a separate dict
    unsigned_quant_act_kwargs = quant_act_kwargs.copy()
    if uint_sym_act_for_unsigned_values and act_quant_format != 'float':
        # In case we support unsigned activation, the output of softmax can be unsigned
        quant_mha_kwargs['attn_output_weights_signed'] = False
        unsigned_quant_act_kwargs['signed'] = False

    # Layerwise is  basic quant kwargs + input_quant
    layerwise_quant_wbiol_kwargs = {**quant_wbiol_kwargs, 'input_quant': act_quant}

    layerwise_quant_mha_kwargs = {**quant_mha_kwargs, 'in_proj_input_quant': act_quant}

    quant_layer_map = {
        torch.nn.Linear: (qnn.QuantLinear, quant_wbiol_kwargs),
        torch.nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs),
        torch.nn.Conv1d: (qnn.QuantConv1d, quant_wbiol_kwargs),
        torch.nn.Conv2d: (qnn.QuantConv2d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose1d: (qnn.QuantConvTranspose1d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose2d: (qnn.QuantConvTranspose2d, quant_wbiol_kwargs),}

    quant_act_map = {
        torch.nn.ReLU: (qnn.QuantReLU, {
            **unsigned_quant_act_kwargs}),
        torch.nn.ReLU6: (qnn.QuantReLU, {
            **unsigned_quant_act_kwargs}),
        torch.nn.Sigmoid: (qnn.QuantSigmoid, {
            **unsigned_quant_act_kwargs}),}
    quant_identity_map = {
        'signed': (qnn.QuantIdentity, {
            **quant_act_kwargs}),
        'unsigned': (qnn.QuantIdentity, {
            **unsigned_quant_act_kwargs}),}
    quant_layerwise_layer_map = {
        torch.nn.Linear: (qnn.QuantLinear, layerwise_quant_wbiol_kwargs),
        torch.nn.MultiheadAttention: (qnn.QuantMultiheadAttention, layerwise_quant_mha_kwargs),
        torch.nn.Conv1d: (qnn.QuantConv1d, layerwise_quant_wbiol_kwargs),
        torch.nn.Conv2d: (qnn.QuantConv2d, layerwise_quant_wbiol_kwargs),
        torch.nn.ConvTranspose1d: (qnn.QuantConvTranspose1d, layerwise_quant_wbiol_kwargs),
        torch.nn.ConvTranspose2d: (qnn.QuantConvTranspose2d, layerwise_quant_wbiol_kwargs),}

    return quant_layer_map, quant_layerwise_layer_map, quant_act_map, quant_identity_map


def calibrate(calib_loader, model):
    """
    Perform calibration and bias correction, if enabled
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def calibrate_bn(calib_loader, model):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with norm_correction_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_bias_correction(calib_loader, model):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with bias_correction_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_act_equalization(model, calib_loader, layerwise):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    add_mul_node = layerwise
    with torch.no_grad():
        with activation_equalization_mode(model,
                                          alpha=0.5,
                                          layerwise=layerwise,
                                          add_mul_node=add_mul_node):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_gptq(
        calib_loader,
        model,
        act_order=False,
        use_quant_activations=False,
        create_weight_orig=False,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=128):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gptq_class = partial(
            A2GPTQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gptq_class = GPTQ
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(model,
                       act_order=act_order,
                       use_quant_activations=use_quant_activations,
                       create_weight_orig=create_weight_orig,
                       gptq_class=gptq_class) as gptq:
            gptq_model = gptq.model
            for i in tqdm(range(gptq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


def apply_gpfq(
        calib_loader,
        model,
        act_order,
        create_weight_orig=False,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=128):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gpfq_class = partial(
            A2GPFQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gpfq_class = GPFQ
    with torch.no_grad():
        with gpfq_mode(model,
                       create_weight_orig=create_weight_orig,
                       use_quant_activations=True,
                       act_order=act_order,
                       gpfq_class=gpfq_class) as gpfq:
            gpfq_model = gpfq.model
            for i in tqdm(range(gpfq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gpfq_model(images)
                gpfq.update()


def check_positive_int(*args):
    """
    We check that every inputted value is positive, and an integer.
    If it's None, it is skipped.
    """
    for arg in args:
        if arg is None:
            continue
        assert arg > 0.0
        assert not math.isclose(arg, 0.0)
        assert math.isclose(arg % 1, 0.0)
