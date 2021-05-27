import pytest
import os
import numpy as np
import torch
import brevitas.onnx as bo
from brevitas.nn import QuantLinear, QuantConv2d, QuantIdentity
from brevitas.quant import Int16Bias
from finn.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bias_quant", [Int16Bias, None])
@pytest.mark.parametrize("out_features", [3])
@pytest.mark.parametrize("in_features", [4])
@pytest.mark.parametrize("w_bits", [2, 4])
@pytest.mark.parametrize("channel_scaling", [True, False])
@pytest.mark.parametrize("i_bits", [2, 4])
def test_quant_linear(bias, bias_quant, out_features, in_features, w_bits, channel_scaling, i_bits):
    # required to generated quantized inputs, not part of the exported model to test
    quant_inp = QuantIdentity(bit_width=i_bits, return_quant_tensor=True)
    inp_tensor = quant_inp(torch.randn(1, in_features))
    linear = QuantLinear(
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        bias_quant=bias_quant,
        weight_bit_width=w_bits,
        weight_scaling_per_output_channel=channel_scaling)
    linear.eval()
    model = bo.export_finn_onnx(linear, input_t=inp_tensor, export_path='linear.onnx')
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    # the quantized input tensor passed to FINN should be in integer form
    int_inp_array = inp_tensor.int(float_datatype=True).numpy()
    idict = {model.graph.input[0].name: int_inp_array}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    expected = linear(inp_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()


@pytest.mark.parametrize("dw", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bias_quant", [Int16Bias, None])
@pytest.mark.parametrize("in_features", [8])
@pytest.mark.parametrize("in_channels", [4])
@pytest.mark.parametrize("out_channels", [5])
@pytest.mark.parametrize("w_bits", [2, 4])
@pytest.mark.parametrize("channel_scaling", [True, False])
@pytest.mark.parametrize("kernel_size", [3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("i_bits", [2, 4])
def test_quant_conv2d(
        dw, bias, bias_quant, in_features, in_channels, out_channels, w_bits, channel_scaling,
        kernel_size, padding, stride, i_bits):
    # required to generated quantized inputs, not part of the exported model to test
    quant_inp = QuantIdentity(bit_width=i_bits, return_quant_tensor=True)
    inp_tensor = quant_inp(torch.randn(1, in_channels, in_features, in_features))
    conv = QuantConv2d(
        in_channels=in_channels,
        out_channels=in_channels if dw else out_channels,
        groups=in_channels if dw else 1,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias,
        bias_quant=bias_quant,
        weight_bit_width=w_bits,
        weight_scaling_per_output_channel=channel_scaling)
    conv.eval()
    model = bo.export_finn_onnx(conv, input_t=inp_tensor)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    # the quantized input tensor passed to FINN should be in integer form
    int_inp_array = inp_tensor.int(float_datatype=True).numpy()
    idict = {model.graph.input[0].name: int_inp_array}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    expected = conv(inp_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()


