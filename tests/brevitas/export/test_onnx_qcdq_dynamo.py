# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import onnx
from packaging import version
import pytest
import torch
from torch import nn

from brevitas import torch_version
from brevitas.export import export_onnx_qcdq
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from tests.marker import jit_disabled_for_export

from ...export_fixture import rm_onnx

pytestmark = pytest.mark.skipif(
    torch_version < version.parse('2.8'), reason='QCDQ dynamo export requires PyTorch >= 2.8')


def _linear(**kwargs):
    layer_kwargs = {
        'input_quant': Int8ActPerTensorFloat,
        'weight_quant': Int8WeightPerTensorFloat,
        'bias_quant': Int32Bias}
    layer_kwargs.update(kwargs)
    return qnn.QuantLinear(8, 8, bias=True, **layer_kwargs)


@jit_disabled_for_export()
def test_dynamo_integer_weight_and_bias_export():
    model = _linear().eval()
    outfile = 'qcdq_dynamo_int_weight_bias.onnx'
    try:
        export_onnx_qcdq(
            model, torch.randn(2, 8), export_path=outfile, dynamo=True, export_weight_q_node=False)
        graph = onnx.load(outfile).graph
    finally:
        rm_onnx(outfile)
    ops = [n.op_type for n in graph.node]
    # Only the input is quantized in the graph: weight and bias are integer initializers
    assert ops.count('QuantizeLinear') == 1
    int_types = {
        i.data_type
        for i in graph.initializer
        if i.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.INT32) and len(i.dims) > 0}
    assert int_types == {onnx.TensorProto.INT8, onnx.TensorProto.INT32}


@jit_disabled_for_export()
@pytest.mark.parametrize('export_weight_q_node', [True, False])
def test_dynamo_shared_quantizer_integer_export_raises(export_weight_q_node):
    first = _linear()
    second = _linear(weight_quant=first.weight_quant, bias_quant=first.bias_quant)
    model = nn.Sequential(first, second).eval()
    with pytest.raises(RuntimeError, match='shared by several layers'):
        export_onnx_qcdq(
            model,
            torch.randn(2, 8),
            export_path='qcdq_dynamo_shared.onnx',
            dynamo=True,
            export_weight_q_node=export_weight_q_node)
