# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import norm_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import COMPUTE_LAYER_MAP
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import QUANT_ACT_MAP
from brevitas.graph.quantize import QUANT_IDENTITY_MAP
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import FLEXML_COMPUTE_LAYER_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_ACT_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_IDENTITY_MAP
from brevitas.graph.target.flexml import quantize_flexml
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import learned_round_iterator
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import save_inp_out_data
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import split_layers

LAYER_MAP = {
    'generic': [COMPUTE_LAYER_MAP, QUANT_ACT_MAP, QUANT_IDENTITY_MAP],
    'layerwise': [LAYERWISE_COMPUTE_LAYER_MAP],
    'flexml': [FLEXML_COMPUTE_LAYER_MAP, FLEXML_QUANT_ACT_MAP, FLEXML_QUANT_IDENTITY_MAP]}

ASYMMETRIC_ACT_QUANT_MAP = {
    'generic': ShiftedUint8ActPerTensorFloat,
    'layerwise': ShiftedUint8ActPerTensorFloat,
    'flexml': ShiftedUint8ActPerTensorFixedPoint}

QUANTIZE_MAP = {'layerwise': layerwise_quantize, 'generic': quantize, 'flexml': quantize_flexml}

BIAS_BIT_WIDTH_MAP = {'int32': Int32Bias, 'int16': Int16Bias}


def quantize_model(
        model,
        backend,
        act_bit_width,
        weight_bit_width,
        layerwise_first_last_bit_width,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_type,
        scale_factor_type,
        weight_narrow_range=False):
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]

    act_quant_asym = None
    if act_quant_type == 'asymmetric':
        act_quant_asym = ASYMMETRIC_ACT_QUANT_MAP[backend]
    maps = [deepcopy(quant_map) for quant_map in LAYER_MAP[backend]]

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
    maps = update_quant_maps(
        maps,
        scale_factor_type=scale_factor_type,
        bias_bit_width=bias_bit_width,
        scaling_per_output_channel=scaling_per_output_channel,
        act_quant_percentile=act_quant_percentile,
        act_quant_asym=act_quant_asym,
        act_bit_width=act_bit_width_or_lambda,
        weight_bit_width=weight_bit_width_or_lambda,
        weight_narrow_range=weight_narrow_range)

    if len(maps) == 3:
        # Generic and flexml requires three mappings for quantization
        quantize_kwargs = {
            'compute_layer_map': maps[0], 'quant_act_map': maps[1], 'quant_identity_map': maps[2]}
    elif len(maps) == 1:
        # Layerwise requires only the compute layer mapping
        quantize_kwargs = {'compute_layer_map': maps[0]}

    quant_model = quantize_fn(model, **quantize_kwargs)
    return quant_model


def update_quant_maps(
        maps,
        scale_factor_type,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_asym,
        act_bit_width,
        weight_bit_width,
        weight_narrow_range):
    """
    Starting from pre-defined quantizers, modify them to match the desired configuration
    """

    act_kwargs = {'bit_width': act_bit_width, 'high_percentile_q': act_quant_percentile}

    weight_kwargs = {
        'scaling_impl_type': ScalingImplType.PARAMETER_FROM_STATS,
        'scaling_per_output_channel': scaling_per_output_channel,
        'bit_width': weight_bit_width,
        'narrow_range': weight_narrow_range}

    scale_factor_dict = {}
    if scale_factor_type == 'po2':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
        scale_factor_dict['restrict_value_float_to_int_impl'] = CeilSte
    elif scale_factor_type == 'float32':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.FP

    act_kwargs.update(scale_factor_dict)
    weight_kwargs.update(scale_factor_dict)

    # In MHA some activations need to be always with symmetric quantizers to avoid costly ops
    act_kwargs_sym_only = deepcopy(act_kwargs)

    # If activation quantization is asymmetric, update it and add low_percentile
    if act_quant_asym is not None:
        act_kwargs['act_quant'] = act_quant_asym
        act_kwargs['low_percentile_q'] = 100.0 - act_quant_percentile

    def weight_kwargs_prefix(prefix):
        return {prefix + k: v for k, v in weight_kwargs.items()}

    def act_kwargs_prefix(prefix, kwargs):
        updated_kwargs = {}
        for k, v in kwargs.items():
            key = k
            if prefix != '':
                key = prefix + key.replace('act_', '')
            updated_kwargs[key] = v
        return updated_kwargs

    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width]
    for map in maps:
        for k, v in map.items():
            if v is None:
                # Non quantized layer, continue
                continue
            quantizer_class, quantizer_kwargs = v
            if issubclass(quantizer_class, QuantWBIOL):
                # Update weight and bias
                map[k][1].update(weight_kwargs_prefix('weight_'))
                map[k][1]['bias_quant'] = bias_quant
                # If we are using asymmetric activations, return_quant_tensor must be False
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                # If input_quant is defined, we need to update it with correct arguments
                if 'input_quant' in quantizer_kwargs.keys():
                    # Add kwargs arguments to input_quant, if present
                    map[k][1].update(act_kwargs_prefix('input_', act_kwargs))
            elif quantizer_class == QuantMultiheadAttention:
                # Update weights and bias
                map[k][1].update(weight_kwargs_prefix('in_proj_weight_'))
                map[k][1].update(weight_kwargs_prefix('out_proj_weight_'))
                map[k][1]['in_proj_bias_quant'] = bias_quant
                map[k][1]['out_proj_bias_quant'] = bias_quant
                # Update inner requantization activations
                map[k][1].update(act_kwargs_prefix('attn_output_weights_', act_kwargs_sym_only))
                map[k][1].update(act_kwargs_prefix('q_scaled_', act_kwargs_sym_only))
                map[k][1].update(act_kwargs_prefix('k_transposed_', act_kwargs_sym_only))
                map[k][1].update(act_kwargs_prefix('v_', act_kwargs_sym_only))
                map[k][1].update(act_kwargs_prefix('out_proj_input_', act_kwargs))
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                # If in_proj_input_quant is defined, we need to update it with correct arguments
                if 'in_proj_input_quant' in quantizer_kwargs.keys():
                    # Add kwargs arguments to input_quant, if present
                    map[k][1].update(act_kwargs_prefix('in_proj_input_', act_kwargs))
            elif 'act_quant' in quantizer_kwargs.keys() or hasattr(quantizer_class, 'act_quant'):
                # Add kwargs argument to activation quantizers.
                quantizer_kwargs.update(act_kwargs_prefix('', act_kwargs))

    return maps


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
