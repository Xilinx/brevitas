# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from functools import partial
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.functional_quant import functional_quantization_mode
from brevitas.graph.functional_quant import prepare_functional_quantization
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.magr import magr_mode
from brevitas.graph.qronos import Qronos
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas_examples.common.axe import a2gpfq_mode
from brevitas_examples.common.axe import a2gptq_mode
from brevitas_examples.common.axe import AXEMixin

from .equalization_fixtures import *


class _FunctionalExpertLinear(torch.nn.Module):
    """Minimal Qwen-style stacked F.linear experts for functional GPTQ tests."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 3, 4))

    def forward(self, x, expert):
        return torch.nn.functional.linear(x, self.weight[expert])


class _FunctionalExpertMatmul(torch.nn.Module):
    """Minimal GPT-OSS-style [expert, input, output] matmul experts."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 4, 3))

    def forward(self, x, expert):
        return x @ self.weight[expert]


class _FunctionalGroupedExperts(torch.nn.Module):
    """Stacked experts dispatched by grouped-MM cumulative token offsets."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 64, 256, dtype=torch.bfloat16))

    def forward(self, x, offsets):
        return torch._grouped_mm(x, self.weight.transpose(-2, -1), offs=offsets)


class _TwoStageFunctionalGroupedExperts(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_up_weight = torch.nn.Parameter(torch.randn(2, 128, 256, dtype=torch.bfloat16))
        self.down_weight = torch.nn.Parameter(torch.randn(2, 32, 64, dtype=torch.bfloat16))

    def forward(self, x, offsets):
        gate_up = torch._grouped_mm(x, self.gate_up_weight.transpose(-2, -1), offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        return torch._grouped_mm(
            torch.nn.functional.silu(gate) * up, self.down_weight.transpose(-2, -1), offs=offsets)


class _ChangingRouteGroupedExperts(_TwoStageFunctionalGroupedExperts):

    def forward(self, x, selected_experts, routing_weights):
        del routing_weights
        selected_experts = selected_experts.flatten()
        order = torch.argsort(selected_experts)
        counts = torch.bincount(selected_experts, minlength=2)
        offsets = counts.cumsum(0).to(torch.int32)
        return super().forward(x[order], offsets)


class _ChangingRouteGroupedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.experts = _ChangingRouteGroupedExperts()
        self.forward_count = 0

    def forward(self, x):
        self.forward_count += 1
        if self.forward_count % 2:
            selected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device=x.device)
        else:
            selected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=x.device)
        return self.experts(x, selected[:, None], torch.ones(8, 1, device=x.device))


class _FunctionalRoutedExperts(torch.nn.Module):
    """Functional experts with deterministic routed slices for batching parity tests."""

    def __init__(self, num_experts=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_experts, 3, 4))

    def forward(self, x, offsets):
        outputs = []
        start = 0
        for expert, end in enumerate(offsets.tolist()):
            outputs.append(torch.nn.functional.linear(x[start:end], self.weight[expert]))
            start = end
        return torch.cat(outputs)


class _MixedFunctionalExperts(torch.nn.Module):
    """An ordinary quantized layer followed by stacked functional experts."""

    def __init__(self):
        super().__init__()
        self.input_proj = qnn.QuantLinear(4, 4, bias=False, weight_quant=Int8WeightPerTensorFloat)
        self.weight = torch.nn.Parameter(torch.randn(2, 3, 4))

    def forward(self, x, expert):
        return torch.nn.functional.linear(self.input_proj(x), self.weight[expert])


class _TwoFunctionalOwners(torch.nn.Module):
    """Two stacked owners that must be scheduled in projection dependency order."""

    def __init__(self):
        super().__init__()
        self.first_weight = torch.nn.Parameter(torch.randn(2, 3, 4))
        self.second_weight = torch.nn.Parameter(torch.randn(2, 2, 3))

    def forward(self, x, expert):
        x = torch.nn.functional.linear(x, self.first_weight[expert])
        return torch.nn.functional.linear(x, self.second_weight[expert])


def _functional_weight_spec(output_channel_dim, group_dim):
    return (
        Int8WeightPerTensorFloat, {
            'output_channel_dim': output_channel_dim, 'group_dim': group_dim})


@torch.no_grad()
@pytest.mark.parametrize(
    'model_class,quant_map',
    [
        (
            _FunctionalExpertLinear, {
                torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}),
        (
            _FunctionalExpertMatmul,
            {
                torch.matmul: (None, None, _functional_weight_spec(2, 1)),
                torch.Tensor.matmul: (None, None, _functional_weight_spec(2, 1)),
                torch.Tensor.__matmul__: (None, None, _functional_weight_spec(2, 1))}),],
    ids=['f_linear', 'gpt_oss_matmul'])
def test_functional_gptq_updates_only_observed_expert(model_class, quant_map):
    """GPTQ uses a distinct target per expert and leaves inactive slices at RTN."""
    model = model_class().eval()
    x = torch.randn(8, 4)
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gptq_mode(model, functional_state=state, min_samples=1) as mode:
            for _ in range(mode.num_layers):
                mode.model(x, 0)
                mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    state.cleanup()


@torch.no_grad()
def test_functional_targets_share_one_owner_schedule_step():
    """All expert slices of one stacked owner are calibrated in one replay."""
    model = _FunctionalExpertLinear().eval()
    x = torch.randn(8, 4)
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    with functional_quantization_mode(state):
        with gptq_mode(model, functional_state=state, min_samples=1) as mode:
            assert mode.num_layers == 1
            mode.model(x, 0)
            mode.update()
    state.cleanup()


@torch.no_grad()
def test_functional_target_quantizes_only_its_expert_view():
    """GPTQ target quantization does not invoke the proxy on the stacked owner."""
    model = _FunctionalExpertLinear().eval()
    x = torch.randn(8, 4)
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    with gptq_mode(model, functional_state=state) as mode:
        target = mode.functional_targets[0]
        target.quant_weight()
        assert target.target_quant_proxy is not target.owner.proxy
        assert target.target_quant_holder.weight.shape == (3, 4)
    state.cleanup()


@torch.no_grad()
@pytest.mark.parametrize('act_order', [False, True])
def test_functional_gptq_expert_batch_matches_scalar(act_order):
    """Bounded tensor batching preserves independent scalar expert updates."""
    torch.manual_seed(0)
    base_model = _FunctionalRoutedExperts().eval()
    x = torch.randn(16, 4)
    offsets = torch.tensor([4, 8, 12, 16])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    results = []
    for batch_size in (1, 2):
        model = deepcopy(base_model)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
        with functional_quantization_mode(state):
            with gptq_mode(model,
                           functional_state=state,
                           min_samples=1,
                           act_order=act_order,
                           expert_batch_size=batch_size) as mode:
                mode.model(x, offsets)
                mode.update()
        results.append(model.parametrizations.weight.original.detach().clone())
        state.cleanup()
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_functional_magr_expert_batch_matches_scalar():
    torch.manual_seed(2)
    base_model = _FunctionalRoutedExperts().eval()
    x = torch.randn(16, 4)
    offsets = torch.tensor([4, 8, 12, 16])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    results = []
    for batch_size in (1, 2):
        model = deepcopy(base_model)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
        with functional_quantization_mode(state):
            with magr_mode(model,
                           functional_state=state,
                           min_samples=1,
                           num_steps=2,
                           expert_batch_size=batch_size) as mode:
                mode.model(x, offsets)
                mode.update()
        results.append(model.parametrizations.weight.original.detach().clone())
        state.cleanup()
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_functional_magr_batch_ignores_cached_reference_when_disabled():
    base_model = _FunctionalRoutedExperts(num_experts=2).eval()
    x = torch.randn(8, 4)
    offsets = torch.tensor([4, 8])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    results = []
    for batch_size in (1, 2):
        model = deepcopy(base_model)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
        with functional_quantization_mode(state):
            with magr_mode(model,
                           functional_state=state,
                           create_weight_orig=False,
                           min_samples=1,
                           num_steps=1,
                           expert_batch_size=batch_size) as mode:
                for target in mode.functional_targets:
                    target.weight_orig
                with torch.no_grad():
                    model.parametrizations.weight.original.add_(0.25)
                mode.model(x, offsets)
                mode.update()
        results.append(model.parametrizations.weight.original.detach().clone())
        state.cleanup()
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize('act_order', [False, True])
def test_functional_gpfq_expert_batch_matches_scalar(act_order):
    torch.manual_seed(3)
    base_model = _FunctionalRoutedExperts().eval()
    x = torch.randn(16, 4)
    offsets = torch.tensor([4, 8, 12, 16])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    results = []
    for batch_size in (1, 2):
        model = deepcopy(base_model)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
        with functional_quantization_mode(state):
            with gpfq_mode(model,
                           functional_state=state,
                           min_samples=1,
                           act_order=act_order,
                           expert_batch_size=batch_size) as mode:
                mode.model(x, offsets)
                mode.update()
        results.append(model.parametrizations.weight.original.detach().clone())
        state.cleanup()
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize('act_order', [False, True])
def test_functional_qronos_expert_batch_matches_scalar(act_order):
    torch.manual_seed(10)
    base_model = _FunctionalRoutedExperts(num_experts=2).eval()
    x = torch.randn(8, 4)
    offsets = torch.tensor([4, 8])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    results = []
    for batch_size in (1, 2):
        model = deepcopy(base_model)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
        with functional_quantization_mode(state):
            with gpfq_mode(model,
                           functional_state=state,
                           min_samples=1,
                           act_order=act_order,
                           algorithm_impl=Qronos,
                           expert_batch_size=batch_size) as mode:
                mode.model(x, offsets)
                mode.update()
        results.append(model.parametrizations.weight.original.detach().clone())
        state.cleanup()
    torch.testing.assert_close(results[0], results[1], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_functional_scalar_qronos_restores_failed_expert():
    model = _FunctionalRoutedExperts(num_experts=2).eval()
    x = torch.randn(8, 4)
    offsets = torch.tensor([4, 8])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gpfq_mode(model,
                       functional_state=state,
                       min_samples=1,
                       algorithm_impl=Qronos,
                       expert_batch_size=1) as mode:
            mode.model(x, offsets)
            mode.gpxq_layers['weight[1]'].H.zero_()
            mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    state.cleanup()


@torch.no_grad()
@pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
def test_functional_qronos_two_stage_grouped_reference_pass():
    model = _TwoStageFunctionalGroupedExperts().eval()
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    offsets = torch.tensor([2, 4], dtype=torch.int32)
    weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
    state = prepare_functional_quantization(
        model, {torch._grouped_mm: (None, None, weight_spec)}, example_inputs=(x, offsets))
    with functional_quantization_mode(state):
        with gpfq_mode(model,
                       functional_state=state,
                       min_samples=1,
                       algorithm_impl=Qronos,
                       expert_batch_size=2) as mode:
            assert mode.num_layers == 2
            for _ in range(mode.num_layers):
                mode.model(x, offsets)
                mode.update()
    state.cleanup()


@torch.no_grad()
@pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
def test_functional_qronos_replays_quantized_expert_routes():
    model = _ChangingRouteGroupedModel().eval()
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
    state = prepare_functional_quantization(
        model, {torch._grouped_mm: (None, None, weight_spec)}, example_inputs=(x,))
    with functional_quantization_mode(state):
        with gpfq_mode(model,
                       functional_state=state,
                       min_samples=1,
                       algorithm_impl=Qronos,
                       expert_batch_size=2) as mode:
            for _ in range(mode.num_layers):
                mode.model(x)
                mode.update()
    state.cleanup()


def test_qronos_partial_exposes_batched_update():
    algorithm_impl = partial(Qronos, alpha=1e-5)
    batch_impl = getattr(algorithm_impl.func, 'batched_layer_update', None)
    assert batch_impl is Qronos.batched_layer_update


def test_qronos_stability_guard_rejects_explosive_and_nonfinite_weights():
    reference = torch.ones(3, 2, 2)
    weight = reference.clone()
    weight[1].mul_(101.)
    weight[2, 0, 0] = float('nan')
    assert Qronos.stable_weight_mask(weight, reference).tolist() == [True, False, False]


@torch.no_grad()
def test_functional_rtn_fallback_warning_is_aggregated():

    class FirstExpertOnly(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 3, 4))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight[0])

    model = FirstExpertOnly().eval()
    x = torch.randn(4, 4)
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
    with pytest.warns(UserWarning,
                      match='uses RTN fallback for 3 insufficiently calibrated experts'):
        with functional_quantization_mode(state):
            with gptq_mode(model, functional_state=state, min_samples=1) as mode:
                mode.model(x)
                mode.update()
    state.cleanup()


def test_functional_gptq_expert_batch_size_must_be_positive():
    with pytest.raises(ValueError, match='expert_batch_size must be positive'):
        gptq_mode(nn.Linear(4, 3), expert_batch_size=0)


@torch.no_grad()
def test_functional_gptq_starts_after_ordinary_layers():
    """Functional owner scheduling begins after ordinary GPxQ hooks are exhausted."""
    model = _MixedFunctionalExperts().eval()
    x = torch.randn(8, 4)
    model(x, 0)  # Initialize the ordinary layer's quantizer before GPTQ.
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gptq_mode(model, functional_state=state, min_samples=1) as mode:
            assert mode.num_layers == 2
            mode.model(x, 0)
            mode.update()
            assert mode.active_functional_target is not None
            mode.model(x, 0)
            mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    state.cleanup()


@torch.no_grad()
def test_functional_gptq_schedules_stacked_owners_in_order():
    """One replay per owner updates earlier projections before later ones collect inputs."""
    model = _TwoFunctionalOwners().eval()
    x = torch.randn(8, 4)
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    before_first = model.parametrizations.first_weight.original.detach().clone()
    before_second = model.parametrizations.second_weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gptq_mode(model, functional_state=state, min_samples=1, expert_batch_size=2) as mode:
            assert mode.num_layers == 2
            mode.model(x, 0)
            mode.update()
            assert mode.active_functional_target.owner_id.endswith('second_weight')
            mode.model(x, 0)
            mode.update()
    after_first = model.parametrizations.first_weight.original.detach()
    after_second = model.parametrizations.second_weight.original.detach()
    assert not torch.equal(after_first[0], before_first[0])
    assert not torch.equal(after_second[0], before_second[0])
    torch.testing.assert_close(after_first[1], before_first[1])
    torch.testing.assert_close(after_second[1], before_second[1])
    state.cleanup()


@torch.no_grad()
@pytest.mark.parametrize('expert_batch_size', [1, 2])
def test_functional_gptq_isolates_invalid_hessian(expert_batch_size):
    """One failed expert factorization leaves that slice unchanged without blocking siblings."""
    model = _FunctionalRoutedExperts(num_experts=2).eval()
    x = torch.randn(8, 4)
    offsets = torch.tensor([4, 8])
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gptq_mode(model,
                       functional_state=state,
                       min_samples=1,
                       expert_batch_size=expert_batch_size) as mode:
            mode.model(x, offsets)
            mode.gpxq_layers['weight[1]'].H.fill_(float('nan'))
            mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    state.cleanup()


@torch.no_grad()
@pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
def test_functional_grouped_mm_gptq_updates_each_routed_expert():
    """Grouped-MM observations route each expert's activations to its GPTQ target."""
    model = _FunctionalGroupedExperts().eval()
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    offsets = torch.tensor([3, 8], dtype=torch.int32)
    quant_map = {
        torch._grouped_mm: (None, None, _functional_weight_spec(1, 2)),}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, offsets))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with gptq_mode(model, functional_state=state, min_samples=1) as mode:
            assert mode.num_layers == 1
            for _ in range(mode.num_layers):
                mode.model(x, offsets)
                mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    assert not torch.equal(after[1], before[1])
    state.cleanup()


@torch.no_grad()
@pytest.mark.parametrize(
    'model_class,quant_map',
    [
        (
            _FunctionalExpertLinear, {
                torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}),
        (
            _FunctionalExpertMatmul,
            {
                torch.matmul: (None, None, _functional_weight_spec(2, 1)),
                torch.Tensor.matmul: (None, None, _functional_weight_spec(2, 1)),
                torch.Tensor.__matmul__: (None, None, _functional_weight_spec(2, 1))}),],
    ids=['f_linear', 'gpt_oss_matmul'])
@pytest.mark.parametrize('algorithm', ['gpfq', 'qronos', 'magr'])
def test_functional_gpxq_algorithms_update_observed_expert(model_class, quant_map, algorithm):
    """All non-GPTQ GPxQ variants use the same functional target contract."""
    model = model_class().eval()
    x = torch.randn(8, 4)
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    before = model.parametrizations.weight.original.detach().clone()
    if algorithm == 'gpfq':
        context = gpfq_mode(model, functional_state=state, min_samples=1)
    elif algorithm == 'qronos':
        context = gpfq_mode(model, functional_state=state, min_samples=1, algorithm_impl=Qronos)
    else:
        context = magr_mode(model, functional_state=state, min_samples=1, num_steps=1, alpha=0.01)
    with functional_quantization_mode(state):
        with context as mode:
            for _ in range(mode.num_layers):
                mode.model(x, 0)
                mode.update()
    after = model.parametrizations.weight.original.detach()
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    state.cleanup()


@torch.no_grad()
def test_functional_gptq_insufficient_samples_error():
    """Coverage policy is applied after the scheduled expert had a full pass."""
    model = _FunctionalExpertLinear().eval()
    x = torch.randn(8, 4)
    quant_map = {torch.nn.functional.linear: (None, None, _functional_weight_spec(1, 2))}
    state = prepare_functional_quantization(model, quant_map, example_inputs=(x, 0))
    before = model.parametrizations.weight.original.detach().clone()
    with functional_quantization_mode(state):
        with pytest.raises(RuntimeError, match='has 8 samples'):
            with gptq_mode(model,
                           functional_state=state,
                           min_samples=9,
                           insufficient_samples='error') as mode:
                mode.model(x, 0)
                mode.update()
    torch.testing.assert_close(model.parametrizations.weight.original, before)
    state.cleanup()


def _a2q_layer_filter_fnc(layer: nn.Module) -> bool:
    if isinstance(layer, nn.Conv2d):
        # Skip when columns == 1 (kernel_size=1 and depthwise)
        kernel_size = np.prod(layer.kernel_size)
        if kernel_size == 1 and layer.groups == layer.in_channels:
            return False
    # Known issue with ConvTranspose2d (#1479)
    if isinstance(layer, nn.ConvTranspose2d):
        return False
    return gpxq_mode._is_module_supported(None, layer)


def _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width):
    # Independently recompute the worst-case signed integer accumulator from the final quantized
    # weights and assert it fits the budget. This is the inference-time guarantee AXE exists to
    # provide, checked without relying on AXE's internal accumulator tracking. We reuse the AXE
    # instance's own input bounds and weight unrolling so we check against exactly what it
    # constrained against.
    max_limit = 2 ** (max_accumulator_bit_width - 1) - 1
    input_max, input_min = gpxq_impl.input_max, gpxq_impl.input_min
    tile_size = gpxq_impl.max_accumulator_tile_size
    # weights as integers, unrolled to [OC, columns] the same way AXE does
    weight = gpxq_impl.reshape_gpxq_weights(gpxq_impl.layer.quant_weight().int().float())
    columns = weight.shape[1]
    for i in range(0, columns, tile_size):
        tile = weight[:, i:i + tile_size]
        pos = torch.clamp_min(tile, 0).sum(dim=1)  # [OC]
        neg = torch.clamp_max(tile, 0).sum(dim=1)  # [OC]
        pos_acc = (input_max * pos + input_min * neg).max().item()
        neg_acc = (-(input_min * pos + input_max * neg)).max().item()
        assert pos_acc <= max_limit, f"positive accumulator {pos_acc} exceeds {max_limit}"
        assert neg_acc <= max_limit, f"negative accumulator {neg_acc} exceeds {max_limit}"


@torch.no_grad()
def _dual_optimization_callback(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        algorithm_impl: nn.Module,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    context_manager_kwargs = dict(
        model=model,
        use_quant_activations=use_quant_activations,
        act_order=act_order,
        algorithm_impl=algorithm_impl)
    context_manager = gpfq_mode
    if max_accumulator_bit_width is not None:
        context_manager = a2gpfq_mode
        context_manager_kwargs.update(
            a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    with context_manager(**context_manager_kwargs) as algo:
        algo_model = algo.model
        for _ in range(algo.num_layers):
            for _, (images, _) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                algo_model(images)
            algo.update()
        if max_accumulator_bit_width is not None:
            # gpxq_layers mixes AXE and plain GPxQ instances (layers failing the a2q filter fall
            # back to the base class); only the AXE instances carry accumulator constraints.
            n_verified = 0
            for gpxq_impl in algo.gpxq_layers.values():
                if isinstance(gpxq_impl, AXEMixin):
                    _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width)
                    n_verified += 1
            # guard against silently verifying nothing (e.g. if no layer became an AXE instance)
            assert n_verified > 0, "AXE was enabled but no layer was accumulator-constrained"


def apply_gpfq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=GPFQ,
        max_accumulator_bit_width=max_accumulator_bit_width,
        max_accumulator_tile_size=max_accumulator_tile_size)


def apply_qronos(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    assert max_accumulator_bit_width is None
    assert max_accumulator_tile_size is None
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=Qronos)


@torch.no_grad()
def apply_gptq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    context_manager_kwargs = dict(
        model=model, act_order=act_order, use_quant_activations=use_quant_activations)
    context_manager = gptq_mode
    if max_accumulator_bit_width is not None:
        context_manager = a2gptq_mode
        context_manager_kwargs.update(
            a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    with context_manager(**context_manager_kwargs) as gptq:
        gptq_model = gptq.model
        for _ in range(gptq.num_layers):
            for _, (images, _) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                gptq_model(images)
            gptq.update()
        if max_accumulator_bit_width is not None:
            # gpxq_layers mixes AXE and plain GPxQ instances (layers failing the a2q filter fall
            # back to the base class); only the AXE instances carry accumulator constraints.
            n_verified = 0
            for gpxq_impl in gptq.gpxq_layers.values():
                if isinstance(gpxq_impl, AXEMixin):
                    _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width)
                    n_verified += 1
            # guard against silently verifying nothing (e.g. if no layer became an AXE instance)
            assert n_verified > 0, "AXE was enabled but no layer was accumulator-constrained"


apply_gpxq_func_map = {"gpfq": apply_gpfq, "gptq": apply_gptq, "qronos": apply_qronos}


class TestQronosUpdateBatch:
    """Tests for Qronos.update_batch verifying correct H and G normalization."""

    INP = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    @staticmethod
    def _make_model():
        """Two QuantLinear layers (2→3→4) with hardcoded weights."""

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear_0 = qnn.QuantLinear(
                    2, 3, bias=False, weight_quant=Int8WeightPerTensorFloat)
                self.linear_1 = qnn.QuantLinear(
                    3, 4, bias=False, weight_quant=Int8WeightPerTensorFloat)

            def forward(self, x):
                return self.linear_1(self.linear_0(x))

        model = Model()
        with torch.no_grad():
            model.linear_0.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]]))
            model.linear_1.weight.copy_(
                torch.tensor([[0.5, -0.3, 0.1], [0.2, 0.4, -0.2], [-0.1, 0.3, 0.5],
                              [0.4, -0.1, 0.2]]))
        return model

    @staticmethod
    def _calibrate(model, calib_loader):
        """Run Qronos calibration, return {layer_name: (H, G)} for each layer."""
        results = {}
        with torch.no_grad():
            with gpfq_mode(model, act_order=False, algorithm_impl=Qronos) as algo:
                for _ in range(algo.num_layers):
                    for data, _ in calib_loader:
                        algo.model(data)
                    for name in algo.current_layer.layer_names:
                        layer = algo.gpxq_layers[name]
                        results[name] = (layer.H.clone(), layer.G.clone())
                    algo.update()
        return results

    def _make_loader(self, batch_size):
        dataset = TensorDataset(self.INP, self.INP)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _init_model(self):
        model = self._make_model()
        model.eval()
        model(self.INP)  # collect scaling factors
        return model

    def test_h_and_g_values(self):
        """Verify H and G have the correct analytical values.

        For linear_0 (first layer), the input is the known external input in both the
        quant and float passes (since input_quant=None), so H == G == X @ X.T / N.

        For linear_1, intermediate activations differ between the quant and float passes
        (weights are quantized in one, float in the other), so H != G in general.
        We verify H is symmetric (X̂ @ X̂.T) and that G is non-zero.

        Current convention on dev: G = X @ X̂.T / N (float @ quant.T).
        TODO: If https://github.com/Xilinx/brevitas/pull/1501 is merged, G convention
        changes to X̂ @ X.T / N (quant @ float.T) and this test must be updated.
        """
        results = self._calibrate(self._init_model(), self._make_loader(batch_size=4))

        # linear_0: no input quant, so quant_input == float_input == X
        x = self.INP.t().unsqueeze(0).float()  # [1, in_features, N]
        expected = x.bmm(x.transpose(2, 1)) / 4  # X @ X.T / N
        H0, G0 = results['linear_0']
        torch.testing.assert_close(H0, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(G0, expected, atol=1e-5, rtol=1e-5)

        # linear_1: H should be symmetric, both H and G should be non-zero
        H1, G1 = results['linear_1']
        torch.testing.assert_close(H1, H1.transpose(1, 2), atol=1e-6, rtol=1e-6)
        assert H1.abs().sum() > 0
        assert G1.abs().sum() > 0

    def test_multi_batch_normalization(self):
        """Runs calibration with 1 batch of 4 vs 2 batches of 2 and asserts H and G
        are identical for both layers, verifies that the running-average normalization
        is batch-size invariant."""
        results_single = self._calibrate(self._init_model(), self._make_loader(batch_size=4))
        results_multi = self._calibrate(self._init_model(), self._make_loader(batch_size=2))

        for name in results_single:
            H_s, G_s = results_single[name]
            H_m, G_m = results_multi[name]
            torch.testing.assert_close(H_s, H_m, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(G_s, G_m, atol=1e-5, rtol=1e-5)

    def test_no_inplace_input_mutation(self):
        """Clones the input before each forward pass and asserts it was not modified,
        catching any in-place normalization (e.g. /=) in update_batch that would
        corrupt inputs."""
        model = self._init_model()
        with torch.no_grad():
            with gpfq_mode(model, act_order=False, algorithm_impl=Qronos) as algo:
                for _ in range(algo.num_layers):
                    for data, _ in self._make_loader(batch_size=2):
                        data_before = data.clone()
                        algo.model(data)
                        torch.testing.assert_close(data, data_before)


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
@pytest.mark.parametrize(
    "apply_gpxq_tuple", apply_gpxq_func_map.items(), ids=apply_gpxq_func_map.keys())
@pytest.mark.parametrize("max_accumulator_bit_width", [None, 12, 32])
@pytest.mark.parametrize("max_accumulator_tile_size", [None, 32])
def test_toy_quant_models(
        toy_quant_model,
        act_order,
        use_quant_activations,
        apply_gpxq_tuple,
        max_accumulator_bit_width,
        max_accumulator_tile_size,
        request):

    test_id = request.node.callspec.id
    input_quant = test_id.split('-')[1]

    torch.manual_seed(SEED)

    if (max_accumulator_bit_width is None) and (max_accumulator_tile_size is not None):
        pytest.skip(
            "max_accumulator_tile_size doesn't matter if max_accumulator_bit_width is None.")

    if (max_accumulator_bit_width is not None) and input_quant.startswith("MXFloat"):
        pytest.skip("No support for AXE + Float.")

    name, apply_gpxq = apply_gpxq_tuple

    if (max_accumulator_bit_width is not None) and (name == "qronos"):
        pytest.skip("No support for AXE + Qronos.")

    model_class = toy_quant_model
    model = model_class()

    gpxq_layers = [mod for mod in model.modules() if _a2q_layer_filter_fnc(mod)]
    if max_accumulator_bit_width is not None and not gpxq_layers:
        pytest.skip(f"AXE does not support any modules in {name}.")

    inp = torch.randn(32, *model.input_size)
    model.eval()
    model(inp)  # test forward pass and collect scaling factors
    dataset = TensorDataset(inp, inp)
    calib_loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)

    def _is_value_error_expected():
        # The conditions below only matter for AXE (A2GPxQ); plain GPxQ has no such constraints
        if max_accumulator_bit_width is None:
            return False
        # AXE needs quantized activation metadata to compute the accumulator bounds. With no input
        # quantizer, AXE.quant_metadata is None and A2GPxQ.single_layer_update raises the exception
        if input_quant == 'None':
            return True
        # Same failure for a different reason: leaving activations unquantized during GPxQ means the
        # quantized input metadata is never captured, so AXE.quant_metadata is None
        if not use_quant_activations:
            return True
        # AXE only supports groupwise weight scales for linear layers; the AXEMixin constructor
        # rejects groupwise weight quantization on convolutions
        for mod in gpxq_layers:
            if mod.weight_quant.is_groupwise and isinstance(mod, SUPPORTED_CONV_OP):
                return True
        return False

    if _is_value_error_expected():
        with pytest.raises(ValueError):
            apply_gpxq(
                calib_loader=calib_loader,
                model=model,
                act_order=act_order,
                use_quant_activations=use_quant_activations,
                max_accumulator_bit_width=max_accumulator_bit_width,
                max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        apply_gpxq(
            calib_loader=calib_loader,
            model=model,
            act_order=act_order,
            use_quant_activations=use_quant_activations,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)


@torch.no_grad()
def apply_magr(
        model,
        dataloader,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        alpha=0.1,
        num_steps=10):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with magr_mode(model,
                   group_of_parallel_layers=group_of_parallel_layers,
                   create_weight_orig=create_weight_orig,
                   num_steps=num_steps,
                   alpha=alpha) as magr:
        magr_model = magr.model
        for _, (images, _) in enumerate(dataloader):
            images = images.to(device)
            images = images.to(dtype)
            magr_model(images)
        magr.update()


def test_magr(toy_model, request):
    test_id = request.node.callspec.id

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in test_id:
        inp = torch.randn(32, *IN_SIZE_LINEAR[1:])
    else:
        inp = torch.randn(32, *IN_SIZE_CONV_SMALL[1:])
    model.eval()
    model(inp)  # test forward pass and collect scaling factors
    dataset = TensorDataset(inp, inp)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)

    apply_magr(model, dataloader)


@pytest_cases.parametrize("gpxq_key", ["gptq", "gpfq"])
def test_gpxq_quant_mha(quant_mha_gpxq_model, gpxq_key):
    # GPxQ descends into QuantMultiheadAttention and optimizes its internal projection
    # QuantLinear layers, whose inputs are always in (L, N, E) layout. GPxQ preprocessing
    # (transpose + reshape to [tokens, features]) is permutation-invariant w.r.t. the batch
    # dimension, so this is coverage that GPxQ runs correctly on QuantMHA across PyTorch
    # versions (with and without named-tensor support).
    torch.manual_seed(SEED)

    model_class = quant_mha_gpxq_model
    model = model_class()
    model.eval()

    # DataLoader batches along dim 0; a per-sample tensor of (seq_len, embed_dim) yields
    # batches that both batch_first settings can consume.
    n_samples = 32
    inp = torch.randn(n_samples, MHA_SEQ_LEN, MHA_EMBED_DIM)
    with torch.no_grad():
        model(inp[:MHA_BATCH_SIZE])  # forward pass to collect scaling factors

    dataset = TensorDataset(inp, inp)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=False)

    apply_gpxq = apply_gpxq_func_map[gpxq_key]
    apply_gpxq(calib_loader=dataloader, model=model, act_order=False, use_quant_activations=False)

    with torch.no_grad():
        out = model(inp[:MHA_BATCH_SIZE])
    assert torch.isfinite(out).all()


class _MockAXEMixin(AXEMixin):
    # Minimal AXEMixin host that exposes get_thresholds without a real layer or context manager.
    # We bypass AXEMixin.__init__ (which needs a layer) and set only what get_thresholds reads.
    def __init__(self, max_accumulator_bit_width, max_accumulator_tile_size, input_bit_width):
        self.max_accumulator_bit_width = torch.tensor(float(max_accumulator_bit_width))
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.groups = 1
        self._input_max = 2 ** (input_bit_width - 1) - 1
        self._input_min = -2 ** (input_bit_width - 1)

    @property
    def input_max(self):
        return self._input_max

    @property
    def input_min(self):
        return self._input_min

    @property
    def radius(self):
        # L1-ball radius (the per-tile accumulator budget in the integer domain)
        return (2 ** self.max_accumulator_bit_width - 2) / float(self.input_max - self.input_min)


class TestAXEThresholds:
    # get_thresholds must project the zero-centered integer-domain weights (w / s) onto an L1 ball
    # of the accumulator budget radius, per tile, then rescale into the float domain. Each test
    # builds weights/scales with a known closed-form oracle and compares against get_thresholds.
    #
    # Monolithic 16-bit accumulator, 8-bit signed input -> radius = (2**16 - 2) / 255 ~= 257.
    accumulator_bit_width = 16
    input_bit_width = 8
    eps = 1e-8

    @property
    def radius(self):
        return (2 ** self.accumulator_bit_width - 2) / (2 ** self.input_bit_width - 1)

    @staticmethod
    def _l1_ball_threshold(a, n, radius):
        # Closed-form soft-threshold for a vector of `n` EQUAL nonzero magnitudes `a` projected
        # onto an L1 ball: 0 if already inside (n * a <= radius), else a - radius / n.
        if n * a <= radius:
            return 0.0
        return a - radius / n

    @staticmethod
    def _expand_group_scales(tile_scales, tile_size, in_features):
        # Expand one scale per tile [OC, n_tiles] to per-input-channel [OC, in_features], mirroring
        # how Brevitas expands a compact groupwise scale back to the weight shape: repeat each
        # group's scale across the group, then slice off the padding down to the real in_features
        # (see brevitas.utils.quant_utils.groupwise_dequant_expand).
        out_features, n_tiles = tile_scales.shape
        scales = tile_scales.unsqueeze(-1).expand(out_features, n_tiles, tile_size)
        return scales.reshape(out_features, n_tiles * tile_size)[:, :in_features]

    def _build_equal_magnitude_case(self, out_features, in_features, tile_size, alpha):
        # Every element has integer-domain magnitude |alpha| (constant), alternating sign for zero
        # mean per tile. Each tile gets its own random positive scale (groupwise along the input
        # dim); the float weight is (integer * scale) so get_thresholds recovers |alpha| after w / s.
        # The last tile may be short (ragged), which get_thresholds pads internally. Returns
        # weight/scales [OC, IC] and the closed-form oracle thresholds [1, n_tiles, OC].
        n_tiles = math.ceil(in_features / tile_size)
        last_tile_size = tile_size if in_features % tile_size == 0 else in_features % tile_size
        assert tile_size % 2 == 0 and last_tile_size % 2 == 0, \
            "each tile needs an even width for the alternating-sign zero mean to hold"

        tile_scales = torch.rand(out_features, n_tiles) + self.eps
        scales = self._expand_group_scales(tile_scales, tile_size, in_features)

        int_weight = torch.full((out_features, in_features), float(alpha))
        int_weight[:, 1::2] *= -1  # alternating sign -> zero mean within every (even-width) tile
        weight = int_weight * scales

        expected = tile_scales.clone() * self._l1_ball_threshold(alpha, tile_size, self.radius)
        # the (possibly ragged) last tile has fewer real elements, so its threshold differs
        expected[:, -1] = tile_scales[:, -1] * self._l1_ball_threshold(
            alpha, last_tile_size, self.radius)
        expected = expected.transpose(0, 1).unsqueeze(0)  # [1, n_tiles, OC]
        return weight, scales, expected

    def _run(self, weight, scales, tile_size, expected):
        axe = _MockAXEMixin(self.accumulator_bit_width, tile_size, self.input_bit_width)
        n_tiles = math.ceil(weight.shape[-1] / tile_size)
        # get_thresholds expects [groups, OC/groups, IC]; groups=1 so add a leading singleton dim
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), n_tiles)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)

    # in_features covers a single tile (16), a ragged/padded last tile (24 -> 16 + 8), and multiple
    # full tiles (32 -> 16 + 16). Per-tile random scales exercise the groupwise scale mapping.
    @pytest.mark.parametrize("in_features", [16, 24, 32])
    def test_outside_ball(self, in_features, out_features=2, tile_size=16, offset=10):
        # every tile outside the ball (n * alpha > radius for all tile widths n) -> theta > 0.
        # size alpha off the smallest tile so the short/ragged tile is outside too.
        n = tile_size if in_features % tile_size == 0 else in_features % tile_size
        alpha = self.radius / n + offset
        weight, scales, expected = self._build_equal_magnitude_case(
            out_features, in_features, tile_size, alpha)
        assert (expected > 0).all()  # confirm we actually exercised the projection
        self._run(weight, scales, tile_size, expected)

    @pytest.mark.parametrize("in_features", [16, 24, 32])
    def test_inside_ball(self, in_features, out_features=2, tile_size=16):
        # every tile inside the ball (n * alpha <= radius for all tile widths n) -> theta == 0
        alpha = self.radius / (2 * tile_size)
        weight, scales, expected = self._build_equal_magnitude_case(
            out_features, in_features, tile_size, alpha)
        assert (expected == 0).all()  # confirm the no-shrinkage branch
        self._run(weight, scales, tile_size, expected)

    def test_unequal_magnitudes(self):
        # hand-solved oracle exercising the sort/threshold-search path that equal magnitudes cannot.
        # accumulator_bit_width=5, input_bit_width=2 -> radius = (2**5 - 2) / (2**2 - 1) = 30/3 = 10.
        # tile (w / s) = [8, 4, -1, -11], mean 0, |v| = [8, 4, 1, 11]. Projecting onto radius 10
        # keeps the top two {11, 8}: theta = (11 + 8 - 10) / 2 = 4.5.
        accumulator_bit_width, input_bit_width, tile_size = 5, 2, 4
        s = 0.25
        weight = (torch.tensor([8.0, 4.0, -1.0, -11.0]) * s).view(1, 4)  # [OC=1, IC=4]
        scales = torch.full((1, 4), s)
        expected = torch.tensor(4.5 * s).view(1, 1, 1)  # [1, n_tiles=1, OC=1]
        axe = _MockAXEMixin(accumulator_bit_width, tile_size, input_bit_width)
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), 1)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)

    def test_nonzero_mean(self):
        # hand-solved oracle exercising the zero-centering step (every other case has mean 0).
        # accumulator_bit_width=5, input_bit_width=2 -> radius = (2**5 - 2) / (2**2 - 1) = 30/3 = 10.
        # tile (w / s) = [10, 6, 2, 2], mean 5 -> centered [5, 1, -3, -3], |v| = [5, 1, 3, 3], sum 12
        # > 10. All four survive: theta = (12 - 10) / 4 = 0.5.
        accumulator_bit_width, input_bit_width, tile_size = 5, 2, 4
        s = 0.25
        weight = (torch.tensor([10.0, 6.0, 2.0, 2.0]) * s).view(1, 4)  # [OC=1, IC=4]
        scales = torch.full((1, 4), s)
        expected = torch.tensor(0.5 * s).view(1, 1, 1)  # [1, n_tiles=1, OC=1]
        axe = _MockAXEMixin(accumulator_bit_width, tile_size, input_bit_width)
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), 1)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)
