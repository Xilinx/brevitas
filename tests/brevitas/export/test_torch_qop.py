# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import torch

from brevitas.export import export_torch_qop
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn import QuantMaxPool2d
from brevitas.nn import QuantReLU
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

OUT_CH = 50
IN_CH = 40
TOLERANCE = 1.1
RANDN_MEAN = 1
RANDN_STD = 3

@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_pytorch_quant_conv_export():
    IN_SIZE = (2, IN_CH, IN_CH, IN_CH)
    KERNEL_SIZE = (3, 3)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_bit_width=7,
                input_bit_width=8,
                output_bit_width=8,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.conv.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_pytorch_quant_linear_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                weight_bit_width=7,
                input_bit_width=8,
                output_bit_width=8,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.linear.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_pytorch_quant_linear_bias_quant_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=True,
                weight_quant=Int8WeightPerTensorFloat,
                weight_bit_width=7,
                input_bit_width=8,
                output_bit_width=8,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.linear.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_pytorch_quant_conv_bias_quant_export():
    IN_SIZE = (2, IN_CH, IN_CH, IN_CH)
    KERNEL_SIZE = (3, 3)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_bit_width=7,
                input_bit_width=8,
                output_bit_width=8,
                weight_quant=Int8WeightPerTensorFloat,
                bias_quant=Int16Bias,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.conv.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_quant_act_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act1 = QuantIdentity(
                bit_width=8,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.act2 = QuantReLU(act_quant=Uint8ActPerTensorFloat)

        def forward(self, x):
            return self.act2(self.act1(x))

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.act2.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()

@jit_disabled_for_export()
def test_quant_max_pool2d_export():
    IN_SIZE = (1, 1, IN_CH, IN_CH)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act = QuantIdentity(
                bit_width=8,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.pool = QuantMaxPool2d(
                kernel_size=KERNEL_SIZE, stride=KERNEL_SIZE,
                return_quant_tensor=False)

        def forward(self, x):
            return self.pool(self.act(x))

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.act.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


@requires_pt_ge('9999', 'Darwin')
@jit_disabled_for_export()
def test_func_quant_max_pool2d_export():
    IN_SIZE = (1, 1, IN_CH, IN_CH)
    KERNEL_SIZE = 2

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act1 = QuantIdentity(
                bit_width=8,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.act2 = QuantIdentity(
                bit_width=8,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)

        def forward(self, x):
            x = self.act1(x)
            x = torch.nn.functional.max_pool2d(x, KERNEL_SIZE)
            x = self.act2(x)
            return x

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    inp = torch.randn(IN_SIZE) * RANDN_STD + RANDN_MEAN # New input with bigger range
    brevitas_out = model(inp)
    pytorch_qf_model = export_torch_qop(model, input_t=inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.act2.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()
