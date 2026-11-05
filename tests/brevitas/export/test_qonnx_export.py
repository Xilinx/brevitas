# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnx
import pytest
import torch

from brevitas.export import enable_debug
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn import QuantLSTM
from brevitas.nn import TruncAvgPool2d
from brevitas.quant.scaled_int import Int4WeightPerTensorFloatDecoupled
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int16Bias
from brevitas_examples import imagenet_classification
from tests.marker import jit_disabled_for_export

from ...export_fixture import qonnx_export_fn
from ...export_fixture import rm_onnx
from .quant_module_fixture import *

OUT_CH = 50
IN_CH = 40
TOLERANCE = 1.1


@jit_disabled_for_export()
def test_generic_quant_linear_export(request, qonnx_export_fn):
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    outfile = f'generic_quant_linear_{request.node.callspec.id}.onnx'
    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_generic_decoupled_quant_linear_export(request, qonnx_export_fn):
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                weight_quant=Int4WeightPerTensorFloatDecoupled,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    outfile = f'generic_decoupled_quant_linear_{request.node.callspec.id}.onnx'
    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_a2q_quant_linear_export(request, a2q_weight_act_quantizers, qonnx_export_fn):
    IN_SIZE = (2, IN_CH)

    _, (weight_quant, io_quant) = a2q_weight_act_quantizers

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                input_quant=io_quant,
                output_quant=io_quant,
                weight_quant=weight_quant,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.1, 0.1)

        def forward(self, x):
            return self.linear(x)

    outfile = f'a2q_quant_linear_{request.node.callspec.id}.onnx'
    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_generic_quant_conv_export(request, qonnx_export_fn):
    IN_SIZE = (2, IN_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                bias=True,
                kernel_size=3,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    outfile = f'generic_quant_conv_{request.node.callspec.id}.onnx'
    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_generic_quant_tensor_export(request, qonnx_export_fn):
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.quant_inp = QuantIdentity(return_quant_tensor=True)
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(self.quant_inp(x))

    outfile = f'generic_quant_tensor_{request.node.callspec.id}.onnx'
    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_generic_quant_avgpool_export(request, qonnx_export_fn):
    IN_SIZE = (2, OUT_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.inp_quant = QuantIdentity(return_quant_tensor=True)
            self.pool = TruncAvgPool2d(kernel_size=2, return_quant_tensor=False)

        def forward(self, x):
            return self.pool(self.inp_quant(x))

    inp = torch.randn(IN_SIZE)

    outfile = f'generic_quant_avgpool_{request.node.callspec.id}.onnx'
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_generic_quant_avgpool_export_quant_input(request, qonnx_export_fn):
    IN_SIZE = (2, OUT_CH, IN_CH, IN_CH)
    inp = torch.randn(IN_SIZE)
    model = nn.Sequential(
        QuantIdentity(return_quant_tensor=True),
        TruncAvgPool2d(kernel_size=2, return_quant_tensor=False))
    model(inp)  # collect scale factors
    model.eval()
    outfile = f'generic_quant_avgpool_quant_input_{request.node.callspec.id}.onnx'
    qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)


@jit_disabled_for_export()
def test_debug_brevitas_onnx_export(request, qonnx_export_fn):
    model, cfg = imagenet_classification.model_with_cfg('quant_mobilenet_v1_4b')
    model.eval()
    debug_hook = enable_debug(model, proxy_level=True)
    input_tensor = torch.randn(1, 3, 224, 224)
    outfile = f'generic_debug_{request.node.callspec.id}.onnx'
    qonnx_export_fn(model, input_t=input_tensor, export_path=outfile)
    rm_onnx(outfile)
    model(input_tensor)
    assert debug_hook.values


@jit_disabled_for_export()
@pytest.mark.parametrize(
    'lstm_kwargs',
    [
        pytest.param({}, id='default'),
        pytest.param({'batch_first': True}, id='batch_first'),
        pytest.param({'coupled_input_forget_gates': True}, id='cifg'),
        pytest.param({
            'bidirectional': True, 'num_layers': 2}, id='bidirectional_2_layers')])
def test_quant_lstm_qonnx_export_matches_module(request, qonnx_export_fn, lstm_kwargs):
    # No runtime implements QuantLSTMCell, so each exported cell is checked against the module
    model = QuantLSTM(input_size=IN_CH, hidden_size=OUT_CH, **lstm_kwargs)
    inp = torch.randn(2, 5, IN_CH) if lstm_kwargs.get('batch_first') else torch.randn(5, 2, IN_CH)
    model(inp)  # collect scale factors
    model.eval()
    outfile = f'quant_lstm_{request.node.callspec.id}.onnx'
    onnx_model = qonnx_export_fn(model, inp, export_path=outfile)
    rm_onnx(outfile)

    initializers = {i.name: onnx.numpy_helper.to_array(i) for i in onnx_model.graph.initializer}
    producers = {output: node for node in onnx_model.graph.node for output in node.output}
    cells = [node for node in onnx_model.graph.node if node.op_type == 'QuantLSTMCell']
    layers = [layer for directions in model.layers for layer in directions]
    # One cell per layer and direction
    assert len(cells) == len(layers)
    cell_quantizers = [
        'output',
        'cell_state',
        'input_acc',
        'forget_acc',
        'cell_acc',
        'output_acc',
        'input_sigmoid',
        'forget_sigmoid',
        'cell_tanh',
        'output_sigmoid',
        'hidden_state_tanh']
    for cell, layer in zip(cells, layers):
        assert cell.domain == 'qonnx.custom_op.general'
        assert (len(cell.input), len(cell.output)) == (48, 3)

        expected_attrs = {
            'batch_first': int(layer.cell.batch_first),
            'reverse_input': int(layer.cell.reverse_input),
            'cifg': int(layer.cifg)}
        expected_quant_params = []
        for name in cell_quantizers:
            quant = getattr(layer.cell, f'{name}_quant')
            # output_sigmoid is the only quantizer exported with a *_narrow_range attribute
            narrow_attr = f'{name}_narrow_range' if name == 'output_sigmoid' else f'{name}_narrow'
            expected_attrs[narrow_attr] = int(quant.is_narrow_range)
            expected_attrs[f'{name}_signed'] = int(quant.is_signed)
            expected_attrs[f'{name}_rounding_mode'] = quant.rounding_mode.encode()
            expected_quant_params += [quant.scale(), quant.zero_point(), quant.bit_width()]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in cell.attribute}
        assert attrs == expected_attrs
        # The last 33 inputs are the scale, zero-point and bit-width of each cell quantizer
        for name, expected in zip(cell.input[15:], expected_quant_params):
            np.testing.assert_allclose(
                initializers[name].flatten(), expected.detach().numpy().flatten())

        # Inputs 3-14 are the ii, if, ic, io, hi, hf, hc, ho weights and the input, forget, cell
        # and output biases, each fed to a Quant node. With cifg, the forget gate inputs are the
        # input gate parameters, as in _QuantLSTMLayer.forward
        forget_gate_params = layer.input_gate_params if layer.cifg else layer.forget_gate_params
        gates = [
            layer.input_gate_params,
            forget_gate_params,
            layer.cell_gate_params,
            layer.output_gate_params]
        expected_params = [g.input_weight.weight for g in gates]
        expected_params += [g.hidden_weight.weight for g in gates]
        expected_params += [g.bias for g in gates]
        for name, expected in zip(cell.input[3:15], expected_params):
            np.testing.assert_array_equal(
                initializers[producers[name].input[0]], expected.detach().numpy())
