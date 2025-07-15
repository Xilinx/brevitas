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

from ...export_fixture import qonnx_export_fn
from ...export_fixture import rm_onnx


@jit_disabled_for_export()
def test_simple_fp8_export():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat)
    outfile = f'weight_fp8.onnx'
    export_onnx_qcdq(model, torch.randn(1, 3), outfile, export_weight_q_node=True)
    rm_onnx(outfile)
    assert True


@jit_disabled_for_export()
def test_qonnx_simple_fp8_export(request, qonnx_export_fn):
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(
        3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat, input_quant=Fp8e4m3OCPActPerTensorFloat)
    outfile = f'qonnx_act_weight_fp8_{request.node.callspec.id}.onnx'
    qonnx_export_fn(model, torch.randn(1, 3), outfile)
    rm_onnx(outfile)
    assert True


@jit_disabled_for_export()
def test_fp8_export_activation():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(3, 16, input_quant=Fp8e4m3OCPActPerTensorFloat)
    outfile = f'act_fp8.onnx'
    export_onnx_qcdq(model, torch.randn(1, 3), outfile, export_weight_q_node=True)
    rm_onnx(outfile)
    assert True


@jit_disabled_for_export()
def test_fp8_export_export_activation():
    if torch_version < version.parse('2.1.0'):
        pytest.skip(f"OCP FP8 types not supported by {torch_version}")

    model = qnn.QuantLinear(
        3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat, input_quant=Fp8e4m3OCPActPerTensorFloat)
    outfile = 'weight_act_fp8.onnx'
    export_onnx_qcdq(model, torch.randn(1, 3), outfile, export_weight_q_node=True)
    rm_onnx(outfile)
    assert True
