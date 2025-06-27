# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.export import enable_debug
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
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
        TruncAvgPool2d(kernel_size=2, return_quant_tensor=False)
    )
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
