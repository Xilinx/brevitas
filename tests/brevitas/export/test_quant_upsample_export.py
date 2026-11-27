# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util

import numpy as np
import onnx
import pytest
import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor
from tests.marker import jit_disabled_for_export

from ...export_fixture import qcdq_export_fn
from ...export_fixture import qonnx_export_fn
from ...export_fixture import rm_onnx

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


@pytest.fixture(
    params=[
        pytest.param(
            lambda: qnn.QuantUpsample(scale_factor=2, mode='nearest'), id='upsample-nearest'),
        pytest.param(
            lambda: qnn.QuantUpsample(scale_factor=2, mode='bilinear'), id='upsample-bilinear'),
        pytest.param(lambda: qnn.QuantUpsamplingNearest2d(scale_factor=2), id='nearest2d'),
        pytest.param(lambda: qnn.QuantUpsamplingBilinear2d(scale_factor=2), id='bilinear2d')])
def upsample_layer(request):
    return request.param()


def _make_model(upsample_layer):
    return nn.Sequential(qnn.QuantReLU(bit_width=4, return_quant_tensor=True), upsample_layer)


@jit_disabled_for_export()
def test_quant_upsample_qonnx_export(request, qonnx_export_fn, upsample_layer):
    # Regression test for #1673: Dynamo export failed with
    # AttributeError: 'FakeTensor' object has no attribute 'value'.
    inp = torch.randn(1, 2, 4, 4)
    model = _make_model(upsample_layer)
    model(inp)  # collect scale factors
    model.eval()
    outfile = f'quant_upsample_qonnx_{request.node.callspec.id}.onnx'
    qonnx_export_fn(model, inp, export_path=outfile)
    onnx_model = onnx.load(outfile)
    onnx.checker.check_model(onnx_model)
    assert any(node.op_type == 'Resize' for node in onnx_model.graph.node)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_quant_upsample_qcdq_export(request, qcdq_export_fn, upsample_layer):
    # Regression test for #1673: same Dynamo failure on the QCDQ path.
    inp = torch.randn(1, 2, 4, 4)
    model = _make_model(upsample_layer)
    model(inp)  # collect scale factors
    model.eval()
    outfile = f'quant_upsample_qcdq_{request.node.callspec.id}.onnx'
    qcdq_export_fn(model, inp, export_path=outfile)
    onnx_model = onnx.load(outfile)
    onnx.checker.check_model(onnx_model)
    assert any(node.op_type == 'Resize' for node in onnx_model.graph.node)
    if _HAS_ORT:
        # Compare the exported model against the eager module, especially for
        # bilinear modes, whose outputs must stay on the input quantization
        # grid (Div/Round/Mul by the cached input scale).
        import onnxruntime as ort

        with torch.no_grad():
            ref = model(inp)
        ref_value = ref.value if isinstance(ref, QuantTensor) else ref
        sess = ort.InferenceSession(outfile, providers=['CPUExecutionProvider'])
        out = sess.run(None, {sess.get_inputs()[0].name: inp.numpy()})[0]
        np.testing.assert_allclose(out, ref_value.numpy(), atol=1e-5)
    rm_onnx(outfile)


@torch.no_grad()
def test_quant_upsample_plain_tensor_input(upsample_layer):
    # A plain Tensor input used to raise AttributeError on x.value; it now
    # interpolates the plain value instead.
    inp = torch.randn(1, 2, 4, 4)
    out = upsample_layer(inp)
    assert isinstance(out, torch.Tensor) and not isinstance(out, QuantTensor)
    assert out.shape == (1, 2, 8, 8)


@torch.no_grad()
def test_quant_upsample_bilinear_stays_on_grid(upsample_layer):
    if isinstance(upsample_layer, qnn.QuantUpsamplingNearest2d) or (isinstance(
            upsample_layer, qnn.QuantUpsample) and upsample_layer.mode == 'nearest'):
        pytest.skip("Nearest upsampling does not round onto the input grid.")
    inp = torch.randn(1, 2, 4, 4)
    model = _make_model(upsample_layer)
    model(inp)  # collect scale factors
    model.eval()
    out = model(inp)
    assert isinstance(out, QuantTensor)
    grid = out.value / out.scale
    assert torch.allclose(grid, grid.round(), atol=1e-5)
