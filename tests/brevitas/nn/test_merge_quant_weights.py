# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.core.scaling import ParameterFromStatsFromParameterScaling
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn import QuantLinear
from brevitas.nn.utils import merge_quant_weights
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import \
    insert_learned_round_quantizers
from tests.conftest import SEED

IN_FEATURES = 8
OUT_FEATURES = 16

LEARNED_ROUND_OPTIONS = [
    LearnedRoundImplType.HARD_SIGMOID, LearnedRoundImplType.SIGMOID, LearnedRoundImplType.IDENTITY]


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
    """After merging, standard round should preserve the quantised weights, remove learned
    round and its forward hooks, and reset the rounding mode to ROUND."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False)
    model.eval()
    insert_learned_round_quantizers(model, learned_round_param)
    assert model.weight_quant.rounding_mode == "LEARNED_ROUND"

    _randomise_learned_round(model)
    model.eval()

    # Get quantised weights with learned round active
    quant_before = _get_quant_weights(model)
    hooks_before = len(model._forward_hooks)

    # Merge learned round into weights
    x = torch.randn(4, IN_FEATURES)
    merge_quant_weights(model, x)

    # Verify that learned round has been removed
    for module in model.modules():
        assert not isinstance(module, LearnedRoundSte), \
            "LearnedRoundSte should be removed after merge"

    # Verify that the merge's forward hooks were cleaned up
    hooks_after = len(model._forward_hooks)
    assert hooks_after == hooks_before, "Forward hooks were not cleaned up after merge"

    # Verify that the rounding mode has been reset to standard round
    assert isinstance(
        model.weight_quant.tensor_quant.scaling_impl, ParameterFromStatsFromParameterScaling)
    assert model.weight_quant.rounding_mode == "ROUND"

    # The quantised outputs should match
    quant_after = _get_quant_weights(model)
    for name in quant_before:
        assert torch.allclose(quant_before[name], quant_after[name], atol=1e-6), \
            f"Quantised weights differ for {name} after merge"


@pytest.mark.parametrize("learned_round_param", LEARNED_ROUND_OPTIONS)
def test_merge_quant_weights_forward_equivalence(learned_round_param):
    """The model forward output should be identical before and after merging."""
    torch.manual_seed(SEED)
    model = QuantLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=True)
    model.eval()

    insert_learned_round_quantizers(model, learned_round_param)
    _randomise_learned_round(model)

    model.eval()
    x = torch.randn(4, IN_FEATURES)

    with torch.no_grad():
        out_before = model(x).clone()

    merge_quant_weights(model, x)

    with torch.no_grad():
        out_after = model(x)

    assert torch.allclose(out_before, out_after, atol=1e-5), \
        "Model outputs differ after merge"
