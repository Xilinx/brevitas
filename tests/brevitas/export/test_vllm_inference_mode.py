# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.stats import StatsOp
from brevitas.export.inference import quant_inference_mode
from brevitas.export.inference.handler import FloatInferencetHandler
from brevitas.export.inference.handler import FloatWeightInferencetHandler
from brevitas.export.inference.handler import GroupwiseFloatWeightInferenceHandler
from brevitas.export.inference.handler import GroupwiseIntWeightInferenceHandler
from brevitas.export.inference.handler import IntInferenceHandler
from brevitas.export.inference.handler import IntWeightInferencetHandler
from brevitas.export.inference.manager import InferenceManager
from brevitas.export.inference.vLLM.handler import vLLMDynamicPerRowFloatInferenceHandler
from brevitas.export.inference.vLLM.handler import vLLMGroupwiseFloatInferenceHandler
from brevitas.export.inference.vLLM.handler import vLLMGroupwiseIntInferenceHandler
import brevitas.nn as qnn
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import ShiftedUint8ActPerTensorFloat
from brevitas.quant import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas_examples.common.generative.quant_blocks import RuntimeDynamicStatsScaling
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from tests.brevitas.hyp_helper import float_tensor_st
from tests.marker import requires_pt_ge


class FP8e4m3OCPDynamicActPerRowFloat(Fp8e4m3OCPActPerTensorFloat):
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = True
    proxy_class = DynamicActFloatQuantProxyFromInjector
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(*shape[:-1], 1)
    scaling_stats_permute_dims = None
    stats_reduce_dim = 1


class vLLMTestManager(InferenceManager):
    """A test-local manager that mirrors the vLLMExportManager handler list
    without importing vllm (which is only needed for the export/serving path)."""

    handlers = [
        IntInferenceHandler,
        vLLMDynamicPerRowFloatInferenceHandler,
        FloatInferencetHandler,
        IntWeightInferencetHandler,
        FloatWeightInferencetHandler,
        vLLMGroupwiseIntInferenceHandler,
        GroupwiseIntWeightInferenceHandler,
        vLLMGroupwiseFloatInferenceHandler,
        GroupwiseFloatWeightInferenceHandler,]


WEIGHT_QUANTIZERS = {
    'int8': Int8WeightPerTensorFloat,
    'uint8': ShiftedUint8WeightPerTensorFloat,
    'fp8': Fp8e4m3WeightPerTensorFloat,
    'mxint8': MXInt8Weight,
    'mxfloat8': MXFloat8e4m3Weight,}

ACT_QUANTIZERS = {
    'int8': Int8ActPerTensorFloat,
    'uint8': ShiftedUint8ActPerTensorFloat,
    'fp8': Fp8e4m3ActPerTensorFloat,
    'per_row_dynamic_fp8': FP8e4m3OCPDynamicActPerRowFloat,
    'mxint8': MXInt8Act,
    'mxfloat8': MXFloat8e4m3Act,}


@pytest_cases.parametrize('weight_quantizer', WEIGHT_QUANTIZERS.items())
@given(weight=float_tensor_st(shape=(8, 16), max_val=1e10, min_val=-1e10))
@requires_pt_ge('2.1')
def test_vllm_weight(weight, weight_quantizer):
    name, quant = weight_quantizer

    inp = torch.randn(8, 16)
    linear = qnn.QuantLinear(16, 8, weight_quant=quant)
    linear.weight.data = weight
    linear.eval()

    quant_out = linear.quant_weight().value
    with quant_inference_mode(linear, compile=False, export_manager=vLLMTestManager):
        _ = linear(inp)
        inference_out = linear.quant_weight()
    assert torch.allclose(quant_out, inference_out)


@pytest_cases.parametrize('act_quantizer', ACT_QUANTIZERS.items())
@given(inp=float_tensor_st(shape=(8, 16), max_val=1e10, min_val=-1e10))
@requires_pt_ge('2.1')
def test_vllm_act(inp, act_quantizer):
    name, quant = act_quantizer

    if 'mx' in name:
        extra_kwargs = {'group_dim': 1}
    else:
        extra_kwargs = {}
    if config.JIT_ENABLED and 'dynamic' in name:
        pytest.skip("JIT and dynamic quantization not supported")
    identity = qnn.QuantIdentity(quant, **extra_kwargs)
    out = identity(inp)
    identity.eval()

    quant_out = identity(inp)
    with quant_inference_mode(identity, compile=False, export_manager=vLLMTestManager):
        _ = identity(inp)
        inference_out = identity(inp)

    assert torch.allclose(quant_out, inference_out)
