# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch

from brevitas import torch_version
from brevitas.export import export_onnx_qcdq
import brevitas.nn as qnn
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from tests.marker import jit_disabled_for_export


@jit_disabled_for_export()
def test_simple_fp8_export():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat)
    export_onnx_qcdq(model, torch.randn(1, 3), 'weight_fp8.onnx', export_weight_q_node=True)
    assert True


@jit_disabled_for_export()
def test_fp8_export_activation():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(3, 16, input_quant=Fp8e4m3OCPActPerTensorFloat)
    export_onnx_qcdq(model, torch.randn(1, 3), 'act_fp8.onnx', export_weight_q_node=True)
    assert True


@jit_disabled_for_export()
def test_fp8_export_export_activation():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(
        3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat, input_quant=Fp8e4m3OCPActPerTensorFloat)
    export_onnx_qcdq(model, torch.randn(1, 3), 'weight_act_fp8.onnx', export_weight_q_node=True)
    assert True
