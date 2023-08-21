# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import norm_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import quantize_flexml
import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
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
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import \
    blockwise_learned_round_iterator
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import save_inp_out_data
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import split_blockwise
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import split_layerwise

QUANTIZE_MAP = {'layerwise': layerwise_quantize, 'fx': quantize, 'flexml': quantize_flexml}

BIAS_BIT_WIDTH_MAP = {32: Int32Bias, 16: Int16Bias}

WEIGHT_QUANT_MAP = {
    'float': {
        'stats': {
            'per_tensor': {
                'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
            'per_channel': {
                'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat}},
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
                'sym': Int8WeightPerChannelFixedPointMSE}},}}

INPUT_QUANT_MAP = {
    'float': {
        'stats': {
            'per_tensor': {
                'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat}},
        'mse': {
            'per_tensor': {
                'sym': Int8ActPerTensorFloatMSE, 'asym': ShiftedUint8ActPerTensorFloatMSE}}},
    'po2': {
        'stats': {
            'per_tensor': {
                'sym': Int8ActPerTensorFixedPoint, 'asym': ShiftedUint8ActPerTensorFixedPoint},},
        'mse': {
            'per_tensor': {
                'sym': Int8ActPerTensorFixedPointMSE}},}}


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
        layerwise_first_last_bit_width=8,
        weight_narrow_range=False,
        weight_param_method='stats',
        act_param_method='stats',
        weight_quant_type='sym',
        act_quant_granularity='per_tensor',
        dtype=torch.float32):
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]
    weight_scale_type = scale_factor_type
    act_scale_type = scale_factor_type

    weight_quant_granularity = weight_quant_granularity

    def bit_width_fn(module, other_bit_width):
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            return layerwise_first_last_bit_width
        elif isinstance(module, torch.nn.Linear) and module.out_features == 1000:
            return layerwise_first_last_bit_width
        else:
            return other_bit_width

    weight_bit_width_or_lambda = weight_bit_width if backend != 'layerwise' else lambda module: bit_width_fn(
        module, weight_bit_width)
    act_bit_width_or_lambda = act_bit_width if backend != 'layerwise' else lambda module: bit_width_fn(
        module, act_bit_width)
    quant_layer_map, quant_layerwise_layer_map, quant_act_map, quant_identity_map = create_quant_maps(dtype=dtype,
                            bias_bit_width=bias_bit_width,
                            weight_bit_width=weight_bit_width_or_lambda,
                            weight_param_method=weight_param_method,
                            weight_scale_type=weight_scale_type,
                            weight_quant_type=weight_quant_type,
                            weight_quant_granularity=weight_quant_granularity,
                            weight_narrow_range=weight_narrow_range,
                            act_bit_width=act_bit_width_or_lambda,
                            act_scale_type=act_scale_type,
                            act_param_method=act_param_method,
                            act_quant_type=act_quant_type,
                            act_quant_granularity=act_quant_granularity,
                            act_quant_percentile=act_quant_percentile)

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

    # Retrieve base input, weight, and bias quantizers
    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width]
    weight_quant = WEIGHT_QUANT_MAP[weight_scale_type][weight_param_method][
        weight_quant_granularity][weight_quant_type]
    if act_bit_width is not None:
        act_quant = INPUT_QUANT_MAP[act_scale_type][act_param_method][act_quant_granularity][
            act_quant_type]
        # Some activations in MHA should always be symmetric
        sym_act_quant = INPUT_QUANT_MAP[act_scale_type][act_param_method][act_quant_granularity][
            'sym']
        # Linear layers with 2d input should always be per tensor
        per_tensor_act_quant = INPUT_QUANT_MAP[act_scale_type][act_param_method]['per_tensor'][
            act_quant_type]
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
        if act_quant_type == 'asym':
            act_quant = act_quant.let(**{'low_percentile_q': 100 - act_quant_percentile})
    if sym_act_quant is not None:
        sym_act_quant = sym_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': dtype})
    if per_tensor_act_quant is not None:
        per_tensor_act_quant = per_tensor_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': dtype})
        if act_quant_type == 'asym':
            per_tensor_act_quant = per_tensor_act_quant.let(
                **{'low_percentile_q': 100 - act_quant_percentile})

    weight_quant_and_bit_width = {
        'weight_quant': weight_quant, 'weight_bit_width': weight_bit_width}

    quant_wbiol_kwargs = {
        **weight_quant_and_bit_width,
        'dtype': dtype,
        'return_quant_tensor': False,
        'bias_quant': bias_quant}

    # yapf: disable
    quant_mha_kwargs = {
        **kwargs_prefix('in_proj_', weight_quant_and_bit_width),
        **kwargs_prefix('out_proj_', weight_quant_and_bit_width),
        'in_proj_bias_quant': bias_quant,
        'softmax_input_quant': None,
        'attn_output_weights_quant': sym_act_quant,
        'attn_output_weights_bit_width': act_bit_width,
        'attn_output_weights_signed': False,
        'q_scaled_quant': sym_act_quant,
        'q_scaled_bit_width': act_bit_width,
        'k_transposed_quant': sym_act_quant,
        'k_transposed_bit_width': act_bit_width,
        'v_quant': sym_act_quant,
        'v_bit_width': act_bit_width,
        'out_proj_input_quant': act_quant,
        'out_proj_input_bit_width': act_bit_width,
        'out_proj_bias_quant': bias_quant,
        'out_proj_output_quant': None,
        # activation equalization requires packed_in_proj
        # since it supports only self-attention
        'packed_in_proj': True,
        'dtype': dtype,
        'return_quant_tensor': False}
    # yapf: enable

    # Layerwise is  basic quant kwargs + input_quant
    layerwise_quant_wbiol_kwargs = {
        **quant_wbiol_kwargs, 'input_quant': per_tensor_act_quant, 'input_bit_width': act_bit_width}

    layerwise_quant_mha_kwargs = {
        **quant_mha_kwargs,
        'in_proj_input_quant': per_tensor_act_quant,
        'in_proj_input_bit_width': act_bit_width}

    quant_layer_map = {
        torch.nn.Linear: (qnn.QuantLinear, quant_wbiol_kwargs),
        torch.nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs),
        torch.nn.Conv1d: (qnn.QuantConv1d, quant_wbiol_kwargs),
        torch.nn.Conv2d: (qnn.QuantConv2d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose1d: (qnn.QuantConvTranspose1d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose2d: (qnn.QuantConvTranspose2d, quant_wbiol_kwargs),}

    act_quant_and_bit_width = {'act_quant': act_quant, 'bit_width': act_bit_width}
    quant_act_kwargs = {**act_quant_and_bit_width, 'return_quant_tensor': True}
    quant_act_map = {
        torch.nn.ReLU: (qnn.QuantReLU, {
            **quant_act_kwargs, 'signed': False}),
        torch.nn.ReLU6: (qnn.QuantReLU, {
            **quant_act_kwargs, 'signed': False}),
        torch.nn.Sigmoid: (qnn.QuantSigmoid, {
            **quant_act_kwargs, 'signed': False}),}
    quant_identity_map = {
        'signed': (qnn.QuantIdentity, {
            **quant_act_kwargs}),
        'unsigned': (qnn.QuantIdentity, {
            **quant_act_kwargs, 'signed': False}),}
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
    if not layerwise and not hasattr(model, 'graph'):
        model = symbolic_trace(model)
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


def apply_learned_round_learning(
        model,
        dataloader,
        learned_round_type='layerwise',
        learned_round_block_name=None,
        optimizer_class=torch.optim.Adam,
        iters=2000,
        optimizer_kwargs={'lr': 1e-3}):

    if learned_round_type == 'layerwise':
        blocks = split_layerwise(model)
    elif learned_round_type == 'block':
        assert learned_round_block_name is not None, 'learned_round_block_name is required for blockwise learned rounding'
        blocks = split_blockwise(model, learned_round_block_name)
    else:
        raise ValueError('Learned round strategy not supported')
    print(blocks.keys())
    print(len(blocks))
    print(f"Total Iterations per layer {iters}")

    for block, block_loss, learned_round_modules, scale_parameters in blockwise_learned_round_iterator(blocks, iters=iters, learned_round_type=learned_round_type):
        _, all_fp_out = save_inp_out_data(model, block, dataloader, store_inp=True, store_out=True, keep_gpu=True, disable_quant=True)
        all_quant_inp, _ = save_inp_out_data(model, block, dataloader, store_inp=True, store_out=True, keep_gpu=True, disable_quant=False)
        parameters = []
        for module in learned_round_modules:
            parameters.extend(list(module.parameters()))

        if len(scale_parameters) == 0:
            optimize_act = False
        else:
            optimize_act = True
        optimizer = optimizer_class(parameters, **optimizer_kwargs)
        if optimize_act:
            a_optimizer = optimizer_class(scale_parameters, lr=4e-5)
            a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                a_optimizer, T_max=iters, eta_min=0.)
        max_size = len(all_fp_out)
        pbar = tqdm(range(iters), desc='')
        for i in pbar:
            idx = torch.randint(0, max_size, (dataloader.batch_size,))

            quant_inp, fp_out = all_quant_inp[idx], all_fp_out[idx]
            for module in block.modules():
                if isinstance(module, QuantWBIOL):
                    module.train()

            optimizer.zero_grad()
            if optimize_act:
                a_optimizer.zero_grad()
            quant_out = block(quant_inp)
            loss, rec_loss, round_loss, b = block_loss(quant_out, fp_out)

            loss.backward()
            optimizer.step()
            if optimize_act:
                a_optimizer.step()
                a_scheduler.step()
            pbar.set_description(
                "loss = {:.4f}, rec_loss = {:.4f}, round_loss = {:.4f} - b = {:.4f}".format(
                    loss, rec_loss, round_loss, b))
