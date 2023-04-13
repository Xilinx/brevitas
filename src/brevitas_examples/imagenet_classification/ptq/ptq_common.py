# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.backends.cudnn as cudnn

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import COMPUTE_LAYER_MAP
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import QUANT_ACT_MAP
from brevitas.graph.quantize import QUANT_IDENTITY_MAP
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import FLEXML_COMPUTE_LAYER_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_ACT_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_IDENTITY_MAP
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas.graph.target.flexml import quantize_flexml
from brevitas.inject.enum import RestrictValueType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import accuracy
from brevitas_examples.imagenet_classification.utils import AverageMeter
from brevitas_examples.imagenet_classification.utils import generate_dataloader

LAYER_MAP = {
    'generic': [COMPUTE_LAYER_MAP, QUANT_ACT_MAP, QUANT_IDENTITY_MAP],
    'flexml': [FLEXML_COMPUTE_LAYER_MAP, FLEXML_QUANT_ACT_MAP, FLEXML_QUANT_IDENTITY_MAP]}

ASYMMETRIC_ACT_QUANT_MAP = {
    'generic': ShiftedUint8ActPerTensorFloat, 'flexml': ShiftedUint8ActPerTensorFixedPoint}

QUANTIZE_MAP = {'generic': quantize, 'flexml': quantize_flexml}

BIAS_BIT_WIDTH_MAP = {'int32': Int32Bias, 'int16': Int16Bias}


def quantize_model(
        model,
        backend,
        bit_width,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_type,
        scale_factor_type):
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]

    act_quant_asym = None
    if act_quant_type == 'asymmetric':
        act_quant_asym = ASYMMETRIC_ACT_QUANT_MAP[backend]

    layer_map, act_map, quant_identity_map = update_quant_maps(
        LAYER_MAP[backend],
        scale_factor_type,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        bit_width,
        act_quant_asym)

    quantize_kwargs = {
        'quant_identity_map': quant_identity_map,
        'compute_layer_map': layer_map,
        'quant_act_map': act_map}

    quant_model = quantize_fn(model, **quantize_kwargs)
    return quant_model


def update_quant_maps(
        maps,
        scale_factor_type,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        bit_width,
        asymmetric_quant):
    """
    Starting from pre-defined quantizers, modify them to match the desired configuration
    """

    act_kwargs = {'bit_width': bit_width, 'high_percentile_q': act_quant_percentile}

    if asymmetric_quant is not None:
        act_kwargs['act_quant'] = asymmetric_quant
        act_kwargs['low_percentile_q'] = 100.0 - act_quant_percentile

    weight_kwargs = {
        'scaling_per_output_channel': scaling_per_output_channel, 'bit_width': bit_width}

    scale_factor_dict = {}
    if scale_factor_type == 'po2':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
        scale_factor_dict['restrict_value_float_to_int_impl'] = CeilSte
    elif scale_factor_type == 'float32':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.FP

    act_kwargs.update(scale_factor_dict)
    weight_kwargs.update(scale_factor_dict)

    def weight_kwargs_prefix(prefix):
        return {prefix + k: v for k, v in weight_kwargs.items()}

    def act_kwargs_prefix(prefix):
        updated_kwargs = {}
        for k, v in act_kwargs.items():
            key = k
            if prefix != '':
                key = prefix + key.replace('act_', '')
            updated_kwargs[key] = v
        return updated_kwargs

    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width]
    for map in maps:
        for k, v in map.items():
            if v is None:
                continue
            if issubclass(v[0], QuantWBIOL):
                map[k][1].update(weight_kwargs_prefix('weight_'))
                if asymmetric_quant is not None:
                    map[k][1]['return_quant_tensor'] = False
            elif v[0] == QuantMultiheadAttention:
                map[k][1].update(weight_kwargs_prefix('in_proj_'))
                map[k][1].update(weight_kwargs_prefix('out_proj_'))
                map[k][1].update(act_kwargs_prefix('attn_output_weights_'))
                map[k][1].update(act_kwargs_prefix('q_scaled_'))
                map[k][1].update(act_kwargs_prefix('k_transposed_'))
                map[k][1].update(act_kwargs_prefix('v_'))
                map[k][1].update(act_kwargs_prefix('out_proj_input_'))
                if asymmetric_quant is not None:
                    map[k][1]['return_quant_tensor'] = False
            elif 'act_quant' in v[1].keys():
                v[1].update(act_kwargs_prefix(''))

            for quantizer_arg, quantizer_value in v[1].items():
                if 'bias_quant' in quantizer_arg:
                    v[1][quantizer_arg] = bias_quant
    return maps


def calibrate(calib_loader, model, bias_corr=True):
    """
    Perform calibration and bias correction, if enabled
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, target) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                model(images)

        if bias_corr:
            with bias_correction_mode(model):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)


def validate(val_loader, model):
    """
    Run validation on the desired dataset
    """
    top1 = AverageMeter('Acc@1', ':6.2f')

    def print_accuracy(top1, prefix=''):
        print('{}Avg acc@1 {top1.avg:2.3f}'.format(prefix, top1=top1))

    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            target = target.to(device)
            target = target.to(dtype)
            images = images.to(device)
            images = images.to(dtype)

            output = model(images)
            # measure accuracy
            acc1, = accuracy(output, target, stable=True)
            top1.update(acc1[0], images.size(0))

        print_accuracy(top1, 'Total:')
    return top1.avg.cpu().numpy()
