# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import norm_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import quantize_flexml
from brevitas.inject import value
import brevitas.nn as qnn
from brevitas.quant.experimental.float import Fp8e4m3Act
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloatMSE
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
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import learned_round_iterator
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import save_inp_out_data
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import split_layers

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
                    'sym': Int8WeightPerChannelFixedPointMSE}},}},
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
                    'sym': Fp8e4m3WeightPerChannelFloatMSE}}}}}

INPUT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFloatMSE, 'asym': ShiftedUint8ActPerTensorFloatMSE}}},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFixedPoint, 'asym': ShiftedUint8ActPerTensorFixedPoint},
            },
            'mse': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFixedPointMSE}},}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3ActPerTensorFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Fp8e4m3ActPerTensorFloat},}}}}


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
        uint_sym_act_for_unsigned_values=True,
        dtype=torch.float32):
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]
    weight_scale_type = scale_factor_type
    act_scale_type = scale_factor_type

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
        act_bit_width_dict['act_bit_width'] = layerwise_bit_width_fn_act

    elif quant_format == 'int' and backend != 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = act_bit_width

    if quant_format == 'float' and backend == 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = layerwise_bit_width_fn_weight
        act_bit_width_dict['act_bit_width'] = layerwise_bit_width_fn_act
        weight_bit_width_dict['weight_mantissa_bit_width'] = layerwise_bit_width_fn_weight_mantissa
        weight_bit_width_dict['weight_exponent_bit_width'] = layerwise_bit_width_fn_weight_exponent
        act_bit_width_dict['act_mantissa_bit_width'] = layerwise_bit_width_fn_act_mantissa
        act_bit_width_dict['act_exponent_bit_width'] = layerwise_bit_width_fn_act_exponent
    elif quant_format == 'float' and backend != 'layerwise':
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = act_bit_width
        weight_bit_width_dict['weight_mantissa_bit_width'] = weight_mantissa_bit_width
        weight_bit_width_dict['weight_exponent_bit_width'] = weight_exponent_bit_width
        act_bit_width_dict['act_mantissa_bit_width'] = act_mantissa_bit_width
        act_bit_width_dict['act_exponent_bit_width'] = act_exponent_bit_width


    quant_layer_map, quant_layerwise_layer_map, quant_act_map, quant_identity_map = create_quant_maps(dtype=dtype,
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
        act_param_method=None,
        act_quant_type=None,
        act_quant_granularity=None,
        act_quant_percentile=None):
    """
    Starting from pre-defined quantizers, modify them to match the desired configuration
    """

    def kwargs_prefix(prefix, weight_kwargs):
        return {prefix + k: v for k, v in weight_kwargs.items()}

    weight_bit_width_dict = {'bit_width': weight_bit_width}
    if weight_quant_format == 'float':
        weight_bit_width_dict['exponent_bit_width'] = weight_exponent_bit_width
        weight_bit_width_dict['mantissa_bit_width'] = weight_mantissa_bit_width

    act_bit_width_dict = {'bit_width': act_bit_width}
    if act_quant_format == 'float':
        act_bit_width_dict['exponent_bit_width'] = act_exponent_bit_width
        act_bit_width_dict['mantissa_bit_width'] = act_mantissa_bit_width

    # Retrieve base input, weight, and bias quantizers
    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width]
    weight_quant = WEIGHT_QUANT_MAP[weight_quant_format][weight_scale_type][weight_param_method][
        weight_quant_granularity][weight_quant_type]
    weight_quant = weight_quant.let(**weight_bit_width_dict)

    if act_bit_width is not None:
        act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method][
            act_quant_granularity][act_quant_type]
        # Some activations in MHA should always be symmetric
        sym_act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method][
            act_quant_granularity]['sym']
        # Linear layers with 2d input should always be per tensor
        per_tensor_act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method][
            'per_tensor'][act_quant_type]
        act_quant = act_quant.let(**act_bit_width_dict)
        sym_act_quant = sym_act_quant.let(**act_bit_width_dict)
        per_tensor_act_quant = per_tensor_act_quant.let(**act_bit_width_dict)
    else:
        act_quant = None
        sym_act_quant = None
        per_tensor_act_quant = None

    # Modify the weight quantizer based on the arguments passed in
    weight_quant = weight_quant.let(
        **{
            'narrow_range': weight_narrow_range,
            'scaling_impl': ParameterFromStatsFromParameterScaling})
    if weight_quant_type == 'asym':
        weight_quant = weight_quant.let(zero_point_impl=ParameterFromStatsFromParameterZeroPoint)
    if act_quant is not None:
        act_quant = act_quant.let(**{'high_percentile_q': act_quant_percentile, 'dtype': dtype})
        if act_quant_type == 'asym' and act_quant_percentile is not None:
            act_quant = act_quant.let(**{'low_percentile_q': 100 - act_quant_percentile})
    if sym_act_quant is not None:
        sym_act_quant = sym_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': dtype})
    if per_tensor_act_quant is not None:
        per_tensor_act_quant = per_tensor_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': dtype})
        if act_quant_type == 'asym' and act_quant_percentile is not None:
            per_tensor_act_quant = per_tensor_act_quant.let(
                **{'low_percentile_q': 100 - act_quant_percentile})

    weight_quant_dict = {'weight_quant': weight_quant}

    quant_wbiol_kwargs = {
        **weight_quant_dict, 'dtype': dtype, 'return_quant_tensor': False, 'bias_quant': bias_quant}

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
        'return_quant_tensor': False}
    # yapf: enable

    quant_act_kwargs = {'act_quant': act_quant, 'return_quant_tensor': True}
    # For potentially unsigned activations, we create a separate dict
    unsigned_quant_act_kwargs = quant_act_kwargs.copy()
    if uint_sym_act_for_unsigned_values:
        # In case we support unsigned activation, the output of softmax can be unsigned
        quant_mha_kwargs['attn_output_weights_signed'] = False
        unsigned_quant_act_kwargs['signed'] = False

    # Layerwise is  basic quant kwargs + input_quant
    layerwise_quant_wbiol_kwargs = {**quant_wbiol_kwargs, 'input_quant': per_tensor_act_quant}

    layerwise_quant_mha_kwargs = {**quant_mha_kwargs, 'in_proj_input_quant': per_tensor_act_quant}

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
    with torch.no_grad():
        with activation_equalization_mode(model, alpha=0.5, layerwise=layerwise):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_gptq(calib_loader, model, act_order=False):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(model, act_order=act_order, use_quant_activations=False) as gptq:
            gptq_model = gptq.model
            for i in tqdm(range(gptq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


def apply_gpfq(calib_loader, model, act_order, p=1.0, use_gpfa2q=False, accumulator_bit_width=None):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gpfq_mode(model,
                       p=p,
                       use_quant_activations=True,
                       act_order=act_order,
                       use_gpfa2q=use_gpfa2q,
                       accumulator_bit_width=accumulator_bit_width) as gpfq:
            gpfq_model = gpfq.model
            for i in tqdm(range(gpfq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gpfq_model(images)
                gpfq.update()


def apply_learned_round_learning(
        model, dataloader, optimizer_class=torch.optim.Adam, iters=1000, optimizer_lr=1e-1):
    layers = []
    split_layers(model, layers)
    print(f"Total Iterations per layer {iters}")
    print(f"Number of layers {len(layers)}")

    for layer, layer_loss, learned_round_module in learned_round_iterator(layers, iters=iters):
        optimizer = optimizer_class(learned_round_module.parameters(), lr=optimizer_lr)
        _, all_fp_out = save_inp_out_data(model, layer, dataloader, store_inp=False, store_out=True, keep_gpu=True, disable_quant=True)
        all_quant_inp, _ = save_inp_out_data(model, layer, dataloader, store_inp=True, store_out=True, keep_gpu=True, disable_quant=False)
        max_size = len(all_fp_out)
        pbar = tqdm(range(iters), desc='')
        for i in pbar:
            idx = torch.randint(0, max_size, (dataloader.batch_size,))
            quant_inp, fp_out = all_quant_inp[idx], all_fp_out[idx]
            layer.train()

            optimizer.zero_grad()
            quant_out = layer(quant_inp)
            loss, rec_loss, round_loss, b = layer_loss(quant_out, fp_out)

            loss.backward()
            optimizer.step()
            pbar.set_description(
                "loss = {:.4f}, rec_loss = {:.4f}, round_loss = {:.4f}, b = {:.4f}".format(
                    loss, rec_loss, round_loss, b))
