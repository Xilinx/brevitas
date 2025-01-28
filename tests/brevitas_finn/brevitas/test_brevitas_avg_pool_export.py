# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pytest
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
import torch

from brevitas.export import export_qonnx
from brevitas.nn import TruncAvgPool2d
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_activation import QuantReLU

export_onnx_path = "test_brevitas_avg_pool_export.onnx"


@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("input_bit_width", [4, 8, 16])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("idim", [7, 8])
@pytest.mark.parametrize("restrict_scaling_type", ["log_fp", "power_of_two"])
def test_brevitas_avg_pool_export(
        kernel_size,
        stride,
        signed,
        bit_width,
        input_bit_width,
        channels,
        idim,
        restrict_scaling_type,
        request):
    if signed:
        quant_node = QuantIdentity(
            bit_width=input_bit_width,
            restrict_scaling_type=restrict_scaling_type,
            return_quant_tensor=True,
        )
    else:
        quant_node = QuantReLU(
            bit_width=input_bit_width,
            restrict_scaling_type=restrict_scaling_type,
            return_quant_tensor=True,
        )

    quant_avgpool = TruncAvgPool2d(
        kernel_size=kernel_size, stride=stride, bit_width=bit_width, float_to_int_impl_type='floor')
    model_brevitas = torch.nn.Sequential(quant_node, quant_avgpool)

    # determine input
    input_shape = (1, channels, idim, idim)
    inp = torch.randn(input_shape)
    model_brevitas.train()
    model_brevitas(inp)
    model_brevitas.eval()
    model_brevitas(inp)

    # export
    test_id = request.node.callspec.id
    export_path = test_id + '_' + export_onnx_path
    export_qonnx(model_brevitas, export_path=export_path, input_t=inp)
    model = ModelWrapper(export_path)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # reference brevitas output
    ref_output_array = model_brevitas(inp).tensor.detach().numpy()
    # finn output
    idict = {model.graph.input[0].name: inp.detach().numpy()}
    odict = oxe.execute_onnx(model, idict, True)
    finn_output = odict[model.graph.output[0].name]
    # compare outputs
    if restrict_scaling_type == "power_of_two" and kernel_size == 2:
        atol = 1e-8
    else:
        atol = quant_avgpool.trunc_quant.scale().detach().numpy()  # Allow "off-by-1" errors
    assert np.isclose(ref_output_array, finn_output, atol=atol).all()
    # cleanup
    os.remove(export_path)
