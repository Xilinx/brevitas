"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import os

import torch
from torch import nn

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
    print(trace_inputs[1])
    print(f"Saving unet to {output_path} ...")
    export_manager.change_weight_export(True)
    import brevitas.config as config
    config._FULL_STATE_DICT = True
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        exported_program = torch.export.export(
            UnetExportWrapper(pipe.unet), args=(trace_inputs[0],), kwargs=trace_inputs[1])
    config._FULL_STATE_DICT = False
    torch.export.save(exported_program, output_path)
