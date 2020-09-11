import os

import torch
import numpy as np
import pytest
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import gen_finn_dt_tensor

import brevitas.onnx as bo
from brevitas.onnx import FINNManager
from brevitas.nn import QuantAvgPool2d
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor import pack_quant_tensor
from brevitas.core.quant import QuantType


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
    ishape = (1, channels, idim, idim)
    ibw_tensor = torch.Tensor([input_bit_width])

    b_avgpool = QuantAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        bit_width=bit_width,
        quant_type=QuantType.INT)
    # call forward pass manually once to cache scale factor and bitwidth
    input_tensor = torch.from_numpy(np.zeros(ishape)).float()
    scale = np.ones((1, channels, 1, 1))
    output_scale = torch.from_numpy(scale).float()
    input_quant_tensor = QuantTensor(input_tensor, output_scale, ibw_tensor, signed)
    FINNManager.export_onnx(b_avgpool, ishape, export_onnx_path, input_t=input_quant_tensor)
    model = ModelWrapper(export_onnx_path)

    # determine input FINN datatype
    if signed is True:
        prefix = "INT"
    else:
        prefix = "UINT"
    dt_name = prefix + str(input_bit_width)
    dtype = DataType[dt_name]
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # execution with input tensor using integers and scale = 1
    # calculate golden output
    inp = gen_finn_dt_tensor(dtype, ishape)
    input_tensor = torch.from_numpy(inp).float()
    input_quant_tensor = QuantTensor(input_tensor, output_scale, ibw_tensor, signed)
    b_avgpool.eval()
    expected = b_avgpool.forward(input_quant_tensor).tensor.detach().numpy()

    # finn execution
    idict = {model.graph.input[0].name: inp}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    assert (expected == produced).all()

    # execution with input tensor using float and scale != 1
    scale = np.random.uniform(low=0, high=1, size=(1, channels, 1, 1)).astype(np.float32)
    inp_tensor = inp * scale
    input_tensor = torch.from_numpy(inp_tensor).float()
    input_scale = torch.from_numpy(scale).float()
    input_quant_tensor = QuantTensor(input_tensor, input_scale, ibw_tensor, signed)
    # export again to set the scale values correctly
    bo.export_finn_onnx(b_avgpool, ishape, export_onnx_path, input_t=input_quant_tensor)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    b_avgpool.eval()
    expected = b_avgpool.forward(input_quant_tensor).tensor.detach().numpy()
    # finn execution
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]

    assert np.isclose(expected, produced).all()

    os.remove(export_onnx_path)

