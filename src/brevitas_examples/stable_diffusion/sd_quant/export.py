"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import json
import os

from safetensors.torch import save_file
import torch
from torch import nn

from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.quant_layer import QuantNonLinearActLayer
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode


def export_onnx(pipe, trace_inputs, output_dir, export_manager):
    output_path = os.path.join(output_dir, 'unet.onnx')
    print(f"Saving unet to {output_path} ...")
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        torch.onnx.export(pipe.unet, args=trace_inputs, f=output_path)


def handle_quant_param(layer, layer_dict):
    if layer.input_quant.is_quant_enabled:
        input_scale = layer.input_quant.export_handler.symbolic_kwargs[
            'dequantize_symbolic_kwargs']['scale'].data
        input_zp = layer.input_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
            'zero_point'].data
        layer_dict['input_scale'] = input_scale.cpu().numpy().tolist()
        layer_dict['input_scale_shape'] = input_scale.shape
        layer_dict['input_zp'] = input_zp.to(torch.float32).cpu().numpy().tolist()
        layer_dict['input_zp_shape'] = input_zp.shape
        layer_dict['input_zp_dtype'] = str(input_zp.dtype)
    if layer.weight_quant.is_quant_enabled:
        weight_scale = layer.weight_quant.export_handler.symbolic_kwargs[
            'dequantize_symbolic_kwargs']['scale'].data
        weight_zp = layer.weight_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
            'zero_point'].data
        layer_dict['weight_scale'] = weight_scale.cpu().numpy().tolist()
        nelems = layer.weight.shape[0]
        weight_scale_shape = [nelems] + [1] * (layer.weight.data.ndim - 1)
        layer_dict['weight_scale_shape'] = weight_scale_shape
        if torch.sum(weight_zp.to(torch.float32)) != 0.:
            weight_zp = weight_zp - 128.  # apply offset to have signed z
            layer_dict['weight_zp'] = weight_zp.to(torch.float32).cpu().numpy().tolist()
            layer_dict['weight_zp_shape'] = weight_scale_shape
            layer_dict['weight_zp_dtype'] = str(weight_zp.dtype)
    if layer.output_quant.export_handler.symbolic_kwargs is not None:
        output_scale = layer.output_quant.export_handler.symbolic_kwargs[
            'dequantize_symbolic_kwargs']['scale'].data

        layer_dict['output_scale'] = output_scale.cpu().numpy().tolist()
        layer_dict['output_scale_shape'] = output_scale.shape
    return layer_dict


def export_quant_params(model, output_dir, prefix):
    quant_output_path = os.path.join(output_dir, prefix + 'quant_params.json')
    safetensor_output_path = os.path.join(output_dir, prefix + 'params.safetensors')
    print(f"Saving unet to {safetensor_output_path} ...")
    from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
    export_manager = StdQCDQONNXManager
    export_manager.change_weight_export(
        export_weight_q_node=True)  # We're exporting FP weights + quantization parameters
    quant_params = dict()
    state_dict = model.state_dict()
    state_dict = {k: v for (k, v) in state_dict.items() if 'tensor_quant' not in k}
    state_dict = {k: v for (k, v) in state_dict.items() if not k.endswith('.scale.weight')}
    state_dict = {k.replace('.layer.', '.'): v for (k, v) in state_dict.items()}

    handled_quant_layers = set()
    with torch.no_grad(), brevitas_proxy_export_mode(model, export_manager):
        for name, module in model.named_modules():
            if isinstance(module, EqualizedModule):
                if id(module.layer) in handled_quant_layers:
                    raise RuntimeError("This should not happen")
                if isinstance(module.layer, QuantWeightBiasInputOutputLayer):
                    layer_dict = dict()
                    full_name = name
                    smoothquant_param = module.scale.weight

                    layer_dict['smoothquant_mul'] = smoothquant_param.data.cpu().numpy().tolist()
                    layer_dict['smoothquant_mul_shape'] = module.scale.runtime_shape
                    layer_dict = handle_quant_param(module.layer, layer_dict)

                    quant_params[full_name] = layer_dict
                    handled_quant_layers.add(id(module.layer))
                else:
                    layer_dict = dict()
                    full_name = name
                    smoothquant_param = module.scale.weight

                    layer_dict['smoothquant_mul'] = smoothquant_param.data.cpu().numpy().tolist()
                    layer_dict['smoothquant_mul_shape'] = module.scale.runtime_shape
                    quant_params[full_name] = layer_dict
                    handled_quant_layers.add(id(module.layer))
            elif isinstance(
                    module,
                    QuantWeightBiasInputOutputLayer) and id(module) not in handled_quant_layers:
                full_name = name
                layer_dict = dict()
                layer_dict = handle_quant_param(module, layer_dict)
                quant_params[full_name] = layer_dict
                handled_quant_layers.add(id(module))
            elif isinstance(module, QuantNonLinearActLayer):
                full_name = name
                layer_dict = dict()
                act_scale = module.act_quant.export_handler.symbolic_kwargs[
                    'dequantize_symbolic_kwargs']['scale'].data
                act_zp = module.act_quant.export_handler.symbolic_kwargs[
                    'dequantize_symbolic_kwargs']['zero_point'].data
                layer_dict['act_scale'] = act_scale.cpu().numpy().tolist()
                layer_dict['act_scale_shape'] = act_scale.shape
                layer_dict['act_zp'] = act_zp.to(torch.float32).cpu().numpy().tolist()
                layer_dict['act_zp_shape'] = act_zp.shape
                layer_dict['act_zp_dtype'] = str(act_zp.dtype)
                quant_params[full_name] = layer_dict
                handled_quant_layers.add(id(module))
    with open(quant_output_path, 'w') as file:
        json.dump(quant_params, file, indent="  ")
    save_file(state_dict, safetensor_output_path)
