# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

from hypothesis import given
import pytest_cases
from pytest_cases import fixture
import torch
import torch.nn as nn

from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import load_quant_model_mode
from brevitas.inject.enum import RestrictValueType
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
# Use custom implementation of kthvalue as work around to (b)float16 kernel limitations
from brevitas.utils.torch_utils import kthvalue
from tests.brevitas.hyp_helper import float_tensor_random_size_st
from tests.conftest import SEED

torch.manual_seed(SEED)
IN_CH = 8
OUT_CH = 16
BATCH = 1
REFERENCE_SCALES = {
    'int_quant': (0.00935234408825635910, 0.01362917013466358185),
    'fp_quant': (0.00249395845457911491, 0.00363444536924362183),
    'int_po2_quant': (0.015625, 0.015625),
    'fp_po2_quant': (0.001953125, 0.00390625),}
REFERENCE_INP = torch.tensor([[-1.8645, -0.4071, 1.1971]])
REFERENCE_WEIGHTS = torch.tensor([[1.0023, 0.0205, 1.4604], [-0.2918, -1.8218, -0.7010],
                                  [1.4573, -0.9074, -0.2708]])


def compute_quantile(x, q):
    k = int(math.floor(.01 * q * x.numel() + 0.5))
    result = kthvalue(x.abs().view(-1), k=k)[0]
    return result


def reference_implementation_scale_factors_po2(
        x, q=99.999, min_val=torch.tensor(1e-10), int_scale=128.):
    quant = compute_quantile(x, q)
    quant = torch.max(min_val, quant)
    quant_float_to_int = torch.ceil(
        torch.log2(quant / int_scale))  # Float to Int Implementation for PowerOfTwo scale

    scale = torch.pow(torch.tensor(2.), quant_float_to_int)

    return scale


@given(inp=float_tensor_random_size_st())
def test_scale_factors_ptq_calibration_po2(inp):

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)

        def forward(self, x):
            return self.act(x)

    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            model(inp)

    expected_scale = reference_implementation_scale_factors_po2(inp)
    scale = model.act.act_quant.scale()

    assert torch.allclose(expected_scale, scale)


class Fp8e4m3ActPerTensorFixedPoint(Fp8e4m3ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO


QUANTS = {
    'int_quant': Int8ActPerTensorFloat,
    'fp_quant': Fp8e4m3ActPerTensorFloat,
    'int_po2_quant': Int8ActPerTensorFixedPoint,
    'fp_po2_quant': Fp8e4m3ActPerTensorFixedPoint}


@pytest_cases.parametrize("act_quant", QUANTS.items(), ids=QUANTS.keys())
def test_scale_factors_ptq_calibration_reference(act_quant):

    reference, act_quant = act_quant

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantReLU(act_quant=act_quant)
            self.linear_weights = REFERENCE_WEIGHTS
            self.act_1 = qnn.QuantIdentity(act_quant=act_quant)

        def forward(self, x):
            o = self.act(x)
            o = torch.matmul(o, self.linear_weights)
            return self.act_1(o)

    # Reference input
    inp = REFERENCE_INP
    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            model(inp)

    computed_scale = model.act.act_quant.scale(), model.act_1.act_quant.scale()
    reference_values = REFERENCE_SCALES[reference]
    assert torch.allclose(computed_scale[0], torch.tensor(reference_values[0]))
    assert torch.allclose(computed_scale[1], torch.tensor(reference_values[1]))


def test_calibration_training_state():

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)

        def forward(self, x):
            return self.act(x)

    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            assert model.act.act_quant.training == True
            assert model.training == False

    assert model.act.act_quant.training == False
    assert model.training == False


class TestBiasCorrection():

    @fixture
    def models(self):

        class MyModel(nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.module_list = nn.ModuleList([
                    nn.Linear(IN_CH, OUT_CH, bias=False), nn.Linear(OUT_CH, OUT_CH, bias=False)])

            def forward(self, inp):
                out_0 = self.module_list[0](inp)
                out_1 = self.module_list[1](out_0)
                return torch.cat((out_0, out_1))

        class MyQuantModel(nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.module_list = nn.ModuleList([
                    qnn.QuantLinear(IN_CH, OUT_CH, bias=False, output_quant=Int8ActPerTensorFloat),
                    qnn.QuantLinear(OUT_CH, OUT_CH, bias=False,
                                    output_quant=Int8ActPerTensorFloat)])

            def forward(self, inp):
                out_0 = self.module_list[0](inp)
                out_1 = self.module_list[1](out_0)
                return torch.cat((out_0, out_1))

        quant_model = MyQuantModel()
        model = MyModel()

        quant_model.module_list[0].weight.data = model.module_list[0].weight.data
        quant_model.module_list[1].weight.data = model.module_list[1].weight.data
        model.eval()
        quant_model.eval()

        return model, quant_model

    def test_bias_correction_results(self, models):
        fp_model, quant_model = models
        num_layers = len(quant_model.module_list)

        # Generate 2 random inputs (i.e., batch_size=2)
        inp_list = [torch.randn(BATCH, IN_CH), torch.randn(BATCH, IN_CH)]
        fp_outs = torch.zeros(len(inp_list), num_layers, OUT_CH)
        quant_outs = torch.zeros(len(inp_list), num_layers, OUT_CH)

        error = torch.zeros(num_layers, OUT_CH)

        # Reference Implementation of bias correction
        quant_model.module_list[0].output_quant.disable_quant = True
        quant_model.module_list[1].output_quant.disable_quant = True
        for b, inp in enumerate(inp_list):
            fp_outs[b, :, :] = fp_model(inp)
            quant_outs[b, 0, :] = quant_model.module_list[0](inp)

            quant_outs[b, 1, :] = quant_model.module_list[1](
                fp_outs[b, 0, :])  # The second layer takes as input the "corrected" output
            error += fp_outs[b] - quant_outs[b]
        quant_model.module_list[0].output_quant.disable_quant = False
        quant_model.module_list[1].output_quant.disable_quant = False

        with bias_correction_mode(quant_model):
            assert not quant_model.module_list[0].output_quant.is_quant_enabled
            assert not quant_model.module_list[1].output_quant.is_quant_enabled
            for inp in inp_list:
                quant_model(inp)

        assert quant_model.module_list[0].output_quant.is_quant_enabled
        assert quant_model.module_list[1].output_quant.is_quant_enabled
        assert quant_model.module_list[0].bias is not None
        assert quant_model.module_list[1].bias is not None
        assert torch.allclose(quant_model.module_list[0].bias, error[0] / len(inp_list))
        assert torch.allclose(quant_model.module_list[1].bias, error[1] / len(inp_list))

    def test_bias_correction_hook(self, models):
        fp_model, quant_model = models
        num_layers = len(quant_model.module_list)

        # Generate 2 random inputs (i.e., batch_size=2)
        inp_list = [torch.randn(BATCH, IN_CH), torch.randn(BATCH, IN_CH)]

        inputs = []
        outputs = []

        # If the user tries to modify the output with the forward_hook, this will be ignored
        # because overriden by our own forward_hook
        def simple_hook(mod, inp, out):
            inputs.append(*inp)
            outputs.append(*out)

        fp_outs = torch.zeros(len(inp_list), num_layers, OUT_CH)

        for b, inp in enumerate(inp_list):
            fp_outs[b, :, :] = fp_model(inp)

        quant_model.module_list[1].register_forward_hook(
            simple_hook)  # Register hook on the second layer

        with bias_correction_mode(quant_model):
            for inp in inp_list:
                quant_model(inp)

        assert len(
            outputs
        ) == 2  # Forward hook called only once per input, even though we performed 3 "forwards" per input
        assert (inputs[0] == fp_outs[0, 0, :]).all(
        )  # In bias_correction mode, the input to each layer is equal to the FP output of the previous layer
        assert (inputs[1] == fp_outs[1, 0, :]).all(
        )  # In bias_correction mode, the input to each layer is equal to the FP output of the previous layer


def test_import_bias_correction():

    class SimpleQuantLinearNet(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(qnn.QuantLinear(IN_CH, OUT_CH, bias=False))

        def forward(self, inp):
            return self.net(inp)

    model = SimpleQuantLinearNet()

    with bias_correction_mode(model):
        model(torch.randn((1, IN_CH)))

    for m in model.modules():
        if isinstance(m, qnn.QuantLinear):
            assert m.bias is not None

    new_model = SimpleQuantLinearNet()

    with load_quant_model_mode(new_model):
        new_model.load_state_dict(model.state_dict())

    for m in new_model.modules():
        if isinstance(m, qnn.QuantLinear):
            assert m.bias is not None


def test_bias_correction_flag():

    class SimpleQuantLinearNet(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(qnn.QuantLinear(IN_CH, OUT_CH, bias=False))

        def forward(self, inp):
            return self.net(inp)

    model = SimpleQuantLinearNet()

    with bias_correction_mode(model, skip_if_no_bias=True):
        model(torch.randn((1, IN_CH)))

    for m in model.modules():
        if isinstance(m, qnn.QuantLinear):
            assert m.bias is None
