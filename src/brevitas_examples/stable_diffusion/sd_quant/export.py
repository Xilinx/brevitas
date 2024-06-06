"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import json
import os

import torch
from torch import nn

from brevitas.nn.equalized_layer import EqualizedModule
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


def export_quant_params(pipe, output_dir):
    output_path = os.path.join(output_dir, 'params.json')
    print(f"Saving unet to {output_path} ...")
    from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
    prefix = 'pipe.unet.'
    quant_params = dict()
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, StdQCDQONNXManager):
        for name, module in pipe.unet.named_modules():
            if isinstance(module, EqualizedModule):
                if isinstance(module.layer, QuantWeightBiasInputOutputLayer):
                    layer_dict = dict()
                    full_name = prefix + name
                    smoothquant_param = module.scale.weight
                    input_scale = module.layer.input_quant.export_handler.symbolic_kwargs[
                        'dequantize_symbolic_kwargs']['scale'].data.numpy()
                    input_zp = module.layer.input_quant.export_handler.symbolic_kwargs[
                        'dequantize_symbolic_kwargs']['zero_point'].data.numpy()
                    weight_scale = module.layer.weight_quant.export_handler.symbolic_kwargs[
                        'dequantize_symbolic_kwargs']['scale'].data.numpy()
                    weight_zp = module.layer.weight_quant.export_handler.symbolic_kwargs[
                        'dequantize_symbolic_kwargs']['zero_point'].data.numpy()
                    layer_dict['smoothquant_mul'] = smoothquant_param.data.numpy().tolist()
                    layer_dict['input_scale'] = input_scale.tolist()
                    layer_dict['input_zp'] = input_zp.tolist()
                    layer_dict['weight_scale'] = weight_scale.tolist()
                    layer_dict['weight_zp'] = weight_zp.tolist()
                    quant_params[full_name] = layer_dict
    with open(output_path, 'w') as file:
        json.dump(quant_params, file)
