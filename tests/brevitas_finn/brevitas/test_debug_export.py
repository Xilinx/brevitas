# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.export import enable_debug
from brevitas_examples import bnn_pynq

from ...export_fixture import qonnx_export_fn
from ...export_fixture import rm_onnx

REF_MODEL = 'CNV_2W2A'


def test_debug_finn_onnx_export(request, qonnx_export_fn):
    model, cfg = bnn_pynq.model_with_cfg(REF_MODEL, pretrained=False)
    model.eval()
    debug_hook = enable_debug(model)
    input_tensor = torch.randn(1, 3, 32, 32)
    outfile = f'finn_debug_{request.node.callspec.id}.onnx'
    qonnx_export_fn(model, input_t=input_tensor, export_path=outfile)
    model(input_tensor)
    assert debug_hook.values
    rm_onnx(outfile)
