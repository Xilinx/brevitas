# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from accelerate.hooks import remove_hook_from_module
import pytest
import torch
from torch import nn

from brevitas.nn import QuantLinear
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas_examples.common.accelerate_utils.accelerate import align_input
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks

pytestmark = pytest.mark.llm


def quantized_model():
    return nn.Sequential(
        QuantLinear(
            16, 16, weight_quant=Int8WeightPerTensorFloat, input_quant=Int8ActPerTensorFloat))


def assert_quantization_is_intact(model):
    quant_linear = model[0]
    assert model(torch.ones(1, 16)).shape == (1, 16)
    assert quant_linear.quant_weight().bit_width == 8


def test_remove_hook_from_module_on_quantized_model():
    model = quantized_model()

    # accelerate >= 1.16 removes the attributes it added to a module by probing every value in
    # the module __dict__ with getattr(value, "_accelerate_added_attribute", False). A Brevitas
    # quantizer proxy keeps the injector class of its quantizer in the __dict__, so the attribute
    # lookup on that class has to fail with an AttributeError for the getattr default to be used.
    remove_hook_from_module(model, recurse=True)

    assert_quantization_is_intact(model)


def test_remove_hooks_after_align_input():
    model = quantized_model()
    align_input(model, {"": "cpu"})
    assert hasattr(model, "_hf_hook")

    # This is the code path the LLM examples take when more than one device is used, i.e. a
    # quantized model with an accelerate hook attached is passed back to accelerate.
    remove_hooks(model)

    assert not any(hasattr(module, "_hf_hook") for module in model.modules())
    assert_quantization_is_intact(model)
