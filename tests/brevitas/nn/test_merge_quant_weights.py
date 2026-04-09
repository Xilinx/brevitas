# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn import QuantLinear
from brevitas.nn.utils import merge_quant_weights
from brevitas.quant_tensor import QuantTensor
from tests.conftest import SEED

IN_FEATURES = 8
OUT_FEATURES = 16

LEARNED_ROUND_OPTIONS = [
    LearnedRoundImplType.HARD_SIGMOID, LearnedRoundImplType.SIGMOID, LearnedRoundImplType.IDENTITY]


def _insert_learned_round(model, learned_round_param):
    """Insert learned round quantisers into a model (simplified version for testing)."""
    from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
    for module in model.modules():
        if isinstance(module, QuantWBIOL):
            # Compute init value for the learned round parameter
            if learned_round_param in (LearnedRoundImplType.HARD_SIGMOID,
                                       LearnedRoundImplType.SIGMOID):
                floor_weight = torch.floor(module.weight.data / module.quant_weight().scale)
                delta = (module.weight.data / module.quant_weight().scale) - floor_weight
                value = -torch.log((1.1 - (-0.1)) / (delta - (-0.1)) - 1)
            else:
                value = torch.zeros_like(module.weight.data)

            module.weight_quant.quant_injector = module.weight_quant.quant_injector.let(
                float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
                learned_round_impl_type=learned_round_param,
                learned_round_init=value)
            module.weight_quant.init_tensor_quant(preserve_state_dict=True)


def _get_quant_weights(model):
    """Get the quantised weight outputs for all QuantLinear layers in the model."""
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            quant_weight = module.quant_weight()
            if isinstance(quant_weight, QuantTensor):
                quant_weight = quant_weight.value
            results[name] = quant_weight.detach().clone()
    return results


def _randomise_learned_round(model):
    """Randomise learned round values to simulate training."""
    for module in model.modules():
        if isinstance(module, LearnedRoundSte):
            module.value.data = torch.randn_like(module.value.data)


@pytest.mark.parametrize("learned_round_param", LEARNED_ROUND_OPTIONS)
def test_merge_quant_weights_preserves_quantised_weights(learned_round_param):
    """After merging, standard round should produce the same quantised weights."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False)
    model.eval()

    _insert_learned_round(model, learned_round_param)
    _randomise_learned_round(model)
    model.eval()

    # Get quantised weights with learned round active
    quant_before = _get_quant_weights(model)

    # Merge learned round into weights via context manager
    x = torch.randn(4, IN_FEATURES)
    with merge_quant_weights(model):
        model(x)

    # Verify that learned round has been removed
    for module in model.modules():
        assert not isinstance(module, LearnedRoundSte), \
            "LearnedRoundSte should be removed after merge"

    # The quantised outputs should match
    quant_after = _get_quant_weights(model)
    for name in quant_before:
        assert torch.allclose(quant_before[name], quant_after[name], atol=1e-6), \
            f"Quantised weights differ for {name} after merge"


@pytest.mark.parametrize("learned_round_param", LEARNED_ROUND_OPTIONS)
def test_merge_quant_weights_errors_on_multiple_forward_passes(learned_round_param):
    """Multiple forward passes inside merge_quant_weights should raise RuntimeError."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False)
    model.eval()

    _insert_learned_round(model, learned_round_param)
    _randomise_learned_round(model)
    model.eval()

    x = torch.randn(4, IN_FEATURES)
    with pytest.raises(RuntimeError), merge_quant_weights(model):
        for _ in range(3):
            model(x)


@pytest.mark.parametrize("learned_round_param", LEARNED_ROUND_OPTIONS)
def test_merge_quant_weights_rounding_mode_reset(learned_round_param):
    """After merging, the rounding mode should be ROUND."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False)
    model.eval()

    _insert_learned_round(model, learned_round_param)
    assert model.weight_quant.rounding_mode == "LEARNED_ROUND"

    x = torch.randn(4, IN_FEATURES)
    with merge_quant_weights(model):
        model(x)
    assert model.weight_quant.rounding_mode == "ROUND"


@pytest.mark.parametrize("learned_round_param", LEARNED_ROUND_OPTIONS)
def test_merge_quant_weights_forward_equivalence(learned_round_param):
    """The model forward output should be identical before and after merging."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=True)
    model.eval()

    _insert_learned_round(model, learned_round_param)
    _randomise_learned_round(model)

    model.eval()
    x = torch.randn(4, IN_FEATURES)

    with torch.no_grad():
        out_before = model(x).clone()

    with merge_quant_weights(model):
        model(x)

    with torch.no_grad():
        out_after = model(x)

    assert torch.allclose(out_before, out_after, atol=1e-5), \
        "Model outputs differ after merge"
