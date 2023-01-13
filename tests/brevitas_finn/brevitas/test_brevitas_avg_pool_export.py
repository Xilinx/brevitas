# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
import torch

from brevitas.export import FINNManager
from brevitas.nn import QuantAvgPool2d
from brevitas.quant_tensor import QuantTensor

export_onnx_path = "test_brevitas_avg_pool_export.onnx"


@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("input_bit_width", [4, 8, 16])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("idim", [7, 8])
def test_brevitas_avg_pool_export(
    kernel_size, stride, signed, bit_width, input_bit_width, channels, idim):

    quant_avgpool = QuantAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        bit_width=bit_width)
    quant_avgpool.eval()

    # determine input
    prefix = 'INT' if signed else 'UINT'
    dt_name = prefix + str(input_bit_width)
    dtype = DataType[dt_name]
    input_shape = (1, channels, idim, idim)
    input_array = gen_finn_dt_tensor(dtype, input_shape)
    scale_array = np.random.uniform(low=0, high=1, size=(1, channels, 1, 1)).astype(np.float32)
    input_tensor = torch.from_numpy(input_array * scale_array).float()
    scale_tensor = torch.from_numpy(scale_array).float()
    zp = torch.tensor(0.)
    input_quant_tensor = QuantTensor(
        input_tensor, scale_tensor, zp, input_bit_width, signed, training=False)

    # export
    FINNManager.export(quant_avgpool, export_path=export_onnx_path, input_t=input_quant_tensor)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # reference brevitas output
    ref_output_array = quant_avgpool(input_quant_tensor).tensor.detach().numpy()
    # finn output
    idict = {model.graph.input[0].name: input_array}
    odict = oxe.execute_onnx(model, idict, True)
    finn_output = odict[model.graph.output[0].name]
    # compare outputs
    assert np.isclose(ref_output_array, finn_output).all()
    # cleanup
    os.remove(export_onnx_path)
