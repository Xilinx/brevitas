"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import os

import torch
from torch import nn
from torch._decomp import get_decompositions

from brevitas.backport.fx.experimental.proxy_tensor import make_fx
from brevitas.export.manager import _force_requires_grad_false
from brevitas.export.manager import _JitTraceExportWrapper
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
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
