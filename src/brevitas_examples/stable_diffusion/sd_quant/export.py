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


class UnetExportWrapper(nn.Module):

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs, return_dict=False)


def export_onnx(pipe, trace_inputs, output_dir, export_manager):
    output_path = os.path.join(output_dir, 'unet.onnx')
    print(f"Saving unet to {output_path} ...")
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        torch.onnx.export(pipe.unet, args=trace_inputs, f=output_path)


def export_torch_export(pipe, trace_inputs, output_dir, export_manager):
    output_path = os.path.join(output_dir, 'unet.pt2')
    print(f"Saving unet to {output_path} ...")
    export_manager.change_weight_export(True)
    import brevitas.config as config
    config._FULL_STATE_DICT = True
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        exported_program = torch.export.export(
            UnetExportWrapper(pipe.unet), args=(trace_inputs[0],), kwargs=trace_inputs[1])
    config._FULL_STATE_DICT = False
    print(exported_program.graph.print_tabular())
    torch.export.save(exported_program, output_path)


def handle_quant_param(layer, layer_dict):
    input_scale = layer.input_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
        'scale'].data
    input_zp = layer.input_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
        'zero_point'].data
    weight_scale = layer.weight_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
        'scale'].data
    weight_zp = layer.weight_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
        'zero_point'].data
    if hasattr(layer.output_quant, 'export_handler'):
        output_scale = layer.output_quant.export_handler.symbolic_kwargs[
            'dequantize_symbolic_kwargs']['scale'].data
        output_zp = layer.output_quant.export_handler.symbolic_kwargs['dequantize_symbolic_kwargs'][
            'zero_point'].data
        layer_dict['output_scale'] = output_scale.numpy().tolist()
        layer_dict['output_scale_shape'] = output_scale.shape
        layer_dict['output_zp'] = output_zp.numpy().tolist()
    layer_dict['input_scale'] = input_scale.numpy().tolist()
    layer_dict['input_scale_shape'] = input_scale.shape
    layer_dict['input_zp'] = input_zp.numpy().tolist()
    layer_dict['input_zp_shape'] = input_zp.shape
    layer_dict['weight_scale'] = weight_scale.numpy().tolist()
    layer_dict['weight_scale_shape'] = weight_scale.shape
    layer_dict['weight_zp'] = weight_zp.numpy().tolist()
    layer_dict['weight_zp_shape'] = weight_zp.shape
    return layer_dict


def export_quant_params(pipe, output_dir):
    quant_output_path = os.path.join(output_dir, 'quant_params.json')
    output_path = os.path.join(output_dir, 'params.safetensors')
    print(f"Saving unet to {output_path} ...")
    from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
    quant_params = dict()
    state_dict = pipe.unet.state_dict()
    state_dict = {k: v for (k, v) in state_dict.items() if 'tensor_quant' not in k}
    state_dict = {k: v for (k, v) in state_dict.items() if k.endswith('.scale.weight')}
    handled_quant_layers = set()
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, StdQCDQONNXManager):
        for name, module in pipe.unet.named_modules():
            if isinstance(module, EqualizedModule):
                if id(module.layer) in handled_quant_layers:
                    raise RuntimeError("This should not happen")
                if isinstance(module.layer, QuantWeightBiasInputOutputLayer):
                    layer_dict = dict()
                    full_name = name
                    smoothquant_param = module.scale.weight

                    layer_dict['smoothquant_mul'] = smoothquant_param.data.numpy().tolist()
                    layer_dict['smoothquant_mul_shape'] = module.scale.runtime_shape
                    layer_dict = handle_quant_param(module.layer, layer_dict)

                    quant_params[full_name] = layer_dict
                    handled_quant_layers.add(id(module.layer))
            elif isinstance(
                    module,
                    QuantWeightBiasInputOutputLayer) and id(module) not in handled_quant_layers:
                layer_dict = dict()
                layer_dict = handle_quant_param(module, layer_dict)
                quant_params[full_name] = layer_dict
                handled_quant_layers.add(id(module))
            elif isinstance(module, QuantNonLinearActLayer):
                layer_dict = dict()
                act_scale = module.act_quant.export_handler.symbolic_kwargs[
                    'dequantize_symbolic_kwargs']['scale'].data
                act_zp = module.act_quant.export_handler.symbolic_kwargs[
                    'dequantize_symbolic_kwargs']['zero_point'].data
                layer_dict['act_scale'] = act_scale.numpy().tolist()
                layer_dict['act_scale_shape'] = act_scale.shape
                layer_dict['act_zp'] = act_zp.numpy().tolist()
                quant_params[full_name] = layer_dict
                handled_quant_layers.add(id(module))

    with open(quant_output_path, 'w') as file:
        json.dump(quant_params, file, indent="  ")
    save_file(state_dict, output_path)
