# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch
import torch.nn.utils.parametrize as parametrize

from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.papers.dMX.custom_trainer import remove_bit_width_forward_parametrizations_
from brevitas_examples.papers.dMX.custom_trainer import RotationLearnedBitWidthTrainer
from brevitas_examples.papers.dMX.custom_trainer import WeightFloatBitWidthAverage
from brevitas_examples.papers.dMX.learned_float_quantizer import LearnedFloat
from brevitas_examples.papers.dMX.learned_float_quantizer import MXFP4LearnedbitAct
from brevitas_examples.papers.dMX.learned_float_quantizer import MXFP4LearnedbitWeight
from brevitas_examples.papers.dMX.learned_float_quantizer import MXFP6LearnedFloat


def _quantizers_dict():
    return {"weight_quant": None, "linear_input_quant": None}


def test_learned_float_receives_configured_scaling_minimum():
    configured = LearnedFloat.configure_quantizers_dict(_quantizers_dict(), scaling_min_val=1e-4)

    assert issubclass(configured["weight_quant"], MXFP4LearnedbitWeight)
    assert issubclass(configured["linear_input_quant"], MXFP4LearnedbitAct)
    assert configured["weight_quant"].scaling_min_val == 1e-4
    assert configured["linear_input_quant"].scaling_min_val == 1e-4


def test_learned_float_scaling_minimum_does_not_mutate_registered_injectors():
    first = LearnedFloat.configure_quantizers_dict(_quantizers_dict(), scaling_min_val=1e-4)
    second = LearnedFloat.configure_quantizers_dict(_quantizers_dict(), scaling_min_val=2e-4)

    assert first["weight_quant"] is not second["weight_quant"]
    assert first["linear_input_quant"] is not second["linear_input_quant"]
    assert first["weight_quant"].scaling_min_val == 1e-4
    assert second["weight_quant"].scaling_min_val == 2e-4
    assert MXFP4LearnedbitWeight.scaling_min_val == 1e-10
    assert MXFP4LearnedbitAct.scaling_min_val == 1e-10


def test_mxfp6_learned_float_receives_configured_scaling_minimum():
    configured = MXFP6LearnedFloat.configure_quantizers_dict(
        _quantizers_dict(), scaling_min_val=1e-4)

    assert configured["weight_quant"].scaling_min_val == 1e-4
    assert configured["linear_input_quant"].scaling_min_val == 1e-4


def test_legacy_quantizer_configuration_hook_preserves_override_behavior():

    class StaticQuantizer(BaseQuantizer):
        weight_quant = MXFP4LearnedbitWeight

    configured = StaticQuantizer.configure_quantizers_dict(_quantizers_dict(), scaling_min_val=2e-4)

    assert configured["weight_quant"] is MXFP4LearnedbitWeight
    assert configured["linear_input_quant"] is None


class _RotationAndBitWidth(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.rotation = RotationWeightParametrization(
            rot_mat=torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16)),
            rot_func=lambda tensor,
            matrix,
            _: tensor @ matrix.to(dtype=tensor.dtype),
            axis=1)
        self.bit_width_offset = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))


class _TiedBitWidthOffsets(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Module()
        self.first.bit_width_offset = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))
        self.second = torch.nn.Module()
        self.second.bit_width_offset = self.first.bit_width_offset


def test_fp32_trainable_storage_keeps_rotation_forward_bf16():
    model = _RotationAndBitWidth()
    rotation = model.rotation.rot_mat
    bit_width = model.bit_width_offset
    args = SimpleNamespace(
        tie_vllm_fused_groups=False,
        rotation_parameter_dtype="float32",
        bit_width_parameter_dtype="float32")

    RotationLearnedBitWidthTrainer.prepare_model_for_training(model, args)

    assert model.rotation.rot_mat is rotation
    assert model.parametrizations.bit_width_offset.original is bit_width
    assert rotation.dtype == torch.float32
    assert model.parametrizations.bit_width_offset.original.dtype == torch.float32
    assert model.bit_width_offset.dtype == torch.bfloat16
    assert model.rotation(torch.ones(1, 2, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_fp32_bit_width_master_uses_bf16_forward_view_and_sgd_state():
    model = _RotationAndBitWidth()
    args = SimpleNamespace(
        tie_vllm_fused_groups=False,
        rotation_parameter_dtype=None,
        bit_width_parameter_dtype="float32")

    RotationLearnedBitWidthTrainer.prepare_model_for_training(model, args)

    master = model.parametrizations.bit_width_offset.original
    optimizer = torch.optim.SGD([master], lr=1., momentum=.99)
    loss = model.bit_width_offset.sum()
    loss.backward()
    optimizer.step()

    assert master.dtype == torch.float32
    assert model.bit_width_offset.dtype == torch.bfloat16
    assert master.grad.dtype == torch.float32
    assert optimizer.state[master]["momentum_buffer"].dtype == torch.float32


def test_fp32_bit_width_master_is_shared_by_all_parametrized_owners():
    model = _TiedBitWidthOffsets()
    args = SimpleNamespace(
        tie_vllm_fused_groups=False,
        rotation_parameter_dtype=None,
        bit_width_parameter_dtype="float32")

    RotationLearnedBitWidthTrainer.prepare_model_for_training(model, args)

    first_master = model.first.parametrizations.bit_width_offset.original
    second_master = model.second.parametrizations.bit_width_offset.original

    assert first_master is second_master
    assert first_master.dtype == torch.float32
    assert model.first.bit_width_offset.dtype == torch.bfloat16
    assert model.second.bit_width_offset.dtype == torch.bfloat16


def test_removing_bit_width_parametrizations_restores_original_dtype():
    model = _TiedBitWidthOffsets()
    original = model.first.bit_width_offset
    args = SimpleNamespace(
        tie_vllm_fused_groups=False,
        rotation_parameter_dtype=None,
        bit_width_parameter_dtype="float32")

    RotationLearnedBitWidthTrainer.prepare_model_for_training(model, args)
    remove_bit_width_forward_parametrizations_(model)

    assert model.first.bit_width_offset is original
    assert model.second.bit_width_offset is original
    assert original.dtype == torch.bfloat16
    assert not parametrize.is_parametrized(model.first, "bit_width_offset")
    assert not parametrize.is_parametrized(model.second, "bit_width_offset")


def test_bit_width_average_does_not_downcast_to_model_dtype():
    criterion = WeightFloatBitWidthAverage(torch.nn.Linear(2, 2, dtype=torch.bfloat16))
    criterion.weighted_bit_width_list = [torch.tensor(6.6875), torch.tensor(6.71875)]
    criterion.tot_num_elements = 2

    average = criterion.retrieve()

    assert average.dtype == torch.float32
    assert average.item() == 6.703125
