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


def export_torchscript(pipe, trace_inputs, output_dir, export_manager):
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        fx_g = make_fx(
            UnetExportWrapper(pipe.unet),
            decomposition_table=get_decompositions([
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,]),
        )(*tuple(trace_inputs.values()))
        _force_requires_grad_false(fx_g)
        jit_g = torch.jit.trace(_JitTraceExportWrapper(fx_g), tuple(trace_inputs.values()))
        output_path = os.path.join(output_dir, 'unet.ts')
        print(f"Saving unet to {output_path} ...")
        torch.jit.save(jit_g, output_path)


def export_onnx(pipe, trace_inputs, output_dir, export_manager):
    output_path = os.path.join(output_dir, 'unet.onnx')
    print(f"Saving unet to {output_path} ...")
    with torch.no_grad(), brevitas_proxy_export_mode(pipe.unet, export_manager):
        torch.onnx.export(pipe.unet, args=tuple(trace_inputs.values()), f=output_path)
