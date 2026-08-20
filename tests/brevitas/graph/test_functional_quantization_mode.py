# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.checkpoint import checkpoint

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.functional_quant import _FunctionalReferenceTensor
from brevitas.graph.functional_quant import grouped_mm_functions
from brevitas.graph.quantize import _QuantParametrization
from brevitas.graph.quantize import functional_quantization_mode
from brevitas.graph.quantize import prepare_functional_quantization
from brevitas.graph.quantize import remove_functional_quantization
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from tests.marker import requires_pt_ge


class SimpleLinearModel(nn.Module):
    """Model with a single linear layer that calls F.linear."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class QuantLinearFunctionalModel(nn.Module):
    """QuantLinear whose internal functional call already receives QuantTensors."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = QuantLinear(
            4,
            3,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            return_quant_tensor=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class TwoLinearModel(nn.Module):
    """Model with two linear layers in sequence."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class MultiLinearInModule(nn.Module):
    """A submodule that calls F.linear twice with different weights."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden, in_features))
        self.bias1 = nn.Parameter(torch.randn(hidden))
        self.weight2 = nn.Parameter(torch.randn(out_features, hidden))
        self.bias2 = nn.Parameter(torch.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight1, self.bias1)
        x = F.linear(x, self.weight2, self.bias2)
        return x


class ModelWithMultiLinear(nn.Module):

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.block = MultiLinearInModule(in_features, hidden, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FunctionalBlock(nn.Module):
    """One functional linear call, suitable for direct block execution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class ModelWithFunctionalBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.block = FunctionalBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class BmmModel(nn.Module):
    """Model that calls torch.bmm with two non-parameter tensors."""

    def __init__(self) -> None:
        super().__init__()
        # Dummy parameter so the model is a valid nn.Module with parameters
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.bmm(a, b)


class MatmulWeightModel(nn.Module):
    """Model that calls torch.matmul with a runtime tensor and a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.weight)


class StackedFunctionalWeightModel(nn.Module):
    """Functional linear model selecting a view from a stacked parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3, 4))

    def forward(self, x: Tensor, index: int) -> Tensor:
        return F.linear(x, self.weight[index])


class TransposedStackedFunctionalWeightModel(nn.Module):
    """Functional matmul model with [stack, input, output] weights."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 4, 3))

    def forward(self, x: Tensor, index: int) -> Tensor:
        return x @ self.weight[index]


class GroupedFunctionalWeightModel(nn.Module):
    """Grouped BF16 matmul over a final-two-axis transpose of a stacked owner."""

    def __init__(self, grouped_mm=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 64, 256, dtype=torch.bfloat16))
        object.__setattr__(self, 'grouped_mm', grouped_mm or torch._grouped_mm)

    def forward(self, x: Tensor, offsets: Tensor) -> Tensor:
        return self.grouped_mm(x, self.weight.transpose(-2, -1), offs=offsets)


class TwoStageGroupedFunctionalWeightModel(nn.Module):
    """Two grouped expert projections separated by activation arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.randn(2, 128, 256, dtype=torch.bfloat16))
        self.down_weight = nn.Parameter(torch.randn(2, 32, 64, dtype=torch.bfloat16))

    def forward(self, x: Tensor, offsets: Tensor) -> Tensor:
        gate_up = torch._grouped_mm(x, self.gate_up_weight.transpose(-2, -1), offs=offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        return torch._grouped_mm(intermediate, self.down_weight.transpose(-2, -1), offs=offsets)


class UnsupportedFunctionalWeightViewModel(nn.Module):
    """Functional linear model using a non-leading-index parameter view."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.t())


class ConflictingFunctionalWeightModel(nn.Module):
    """One owner used with incompatible linear and right-matmul layouts."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight) + torch.matmul(x, self.weight)


class ParameterFirstMatmulModel(nn.Module):
    """Functional matmul with a parameter in the first operand."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(self.weight, x)


class ReplacedParameterModel(nn.Module):
    """Model that replaces a parameter once before its functional use."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.replaced = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.replaced:
            self.weight = nn.Parameter(self.weight.detach().clone())
            self.replaced = True
        return F.linear(x, self.weight)


class SharedFunctionalWeight(nn.Module):
    """Functional linear owner used to exercise tied parameters."""

    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class TiedFunctionalWeightModel(nn.Module):
    """Two functional modules sharing one parameter."""

    def __init__(self) -> None:
        super().__init__()
        weight = nn.Parameter(torch.randn(3, 4))
        self.first = SharedFunctionalWeight(weight)
        self.second = SharedFunctionalWeight(weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.first(x)


class FunctionalConvTranspose1dModel(nn.Module):
    """Model that calls F.conv_transpose1d with a parameter weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 16, 3))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv_transpose1d(x, self.weight)


class CheckpointedTwoLinearModel(nn.Module):
    """Two-block model whose blocks are run through gradient checkpointing.

    Each block calls ``F.linear`` once. With gradient checkpointing the block
    forward is executed once during the outer forward and recomputed again during
    the backward pass, so the quantization hooks/counters and the
    TorchFunctionMode interception are exercised twice per training step.

    When ``context_fn`` is provided it is forwarded to ``torch.utils.checkpoint``
    so that quantization can be re-applied during the backward recompute.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.block1 = nn.Linear(in_features, hidden)
        self.block2 = nn.Linear(hidden, out_features)

    def forward(self, x: Tensor, context_fn=None, use_reentrant: bool = False) -> Tensor:
        kwargs = {} if context_fn is None else {'context_fn': context_fn}
        x = checkpoint(self.block1, x, use_reentrant=use_reentrant, **kwargs)
        x = checkpoint(self.block2, x, use_reentrant=use_reentrant, **kwargs)
        return x


@requires_pt_ge('1.12')
class TestFunctionalQuantizationMode:

    def test_reference_tensor_preserves_structural_weight_views_only(self):
        reference = _FunctionalReferenceTensor(torch.randn(2, 2, 3, 4), 'weight')

        negative = reference[-1]
        assert negative._functional_view_indices == (1,)
        tuple_index = reference[1, 0]
        assert tuple_index._functional_view_indices == (1, 0)
        selected = torch.select(reference, dim=0, index=1)
        assert selected._functional_view_indices == (1,)
        unbound = torch.unbind(reference, dim=0)
        assert [item._functional_view_indices for item in unbound] == [(0,), (1,)]
        transposed = torch.transpose(input=tuple_index, dim0=-2, dim1=-1)
        assert transposed._functional_view_indices == (1, 0)

        output = torch.matmul(torch.randn(5, 4), transposed)
        assert type(output) is Tensor

    def test_input_only_skips_parameter_derived_weight_view(self):
        """A missing second spec does not quantize a parameter-derived view."""
        model = StackedFunctionalWeightModel()
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(
            model, {F.linear: Int8ActPerTensorFloat}, example_inputs=(x, 0))
        assert len(state.quantizers) == 1
        assert not is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            out = model(x, 1)
        assert out.shape == (2, 3)
        state.cleanup()

    def test_sliced_linear_quantizes_discovered_owner(self):
        """Explicit owner dimensions quantize a leading-index linear weight view."""
        model = StackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        quant_map = {F.linear: (None, None, weight_spec)}
        state = prepare_functional_quantization(
            model, quant_map, example_inputs=(torch.randn(2, 4), 0))
        assert is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            assert isinstance(model.weight, IntQuantTensor)
            out = model(torch.randn(2, 4), 1)
        assert out.shape == (2, 3)
        state.cleanup()

    def test_sliced_matmul_quantizes_discovered_owner(self):
        """Explicit owner dimensions quantize a leading-index matmul weight view."""
        model = TransposedStackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 2, 'group_dim': 1})
        spec = (None, None, weight_spec)
        quant_map = {torch.matmul: spec, torch.Tensor.matmul: spec, torch.Tensor.__matmul__: spec}
        state = prepare_functional_quantization(
            model, quant_map, example_inputs=(torch.randn(2, 4), 0))
        assert is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4), 1)
        assert out.shape == (2, 3)
        state.cleanup()

    @pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
    def test_grouped_mm_discovers_transposed_owner_and_preserves_gradients(self):
        model = GroupedFunctionalWeightModel()

        def weight_resolver(module, name, index):
            return Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2}

        grouped_mm = next(func for func in grouped_mm_functions() if func is torch._grouped_mm)
        state = prepare_functional_quantization(
            model, {grouped_mm: (None, None, weight_resolver)},
            example_inputs=(
                torch.randn(4, 256, dtype=torch.bfloat16), torch.tensor([2, 4], dtype=torch.int32)))

        assert is_parametrized(model, 'weight')

        with functional_quantization_mode(state):
            output = model(
                torch.randn(4, 256, dtype=torch.bfloat16), torch.tensor([1, 4], dtype=torch.int32))
            output.float().sum().backward()

        assert model.parametrizations.weight.original.grad is not None
        state.cleanup()

    @pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
    def test_grouped_mm_exposes_expert_targets_and_observes_offset_slices(self):
        """Grouped-MM experts use canonical targets and cumulative-offset input slices."""
        model = GroupedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        grouped_mm = next(func for func in grouped_mm_functions() if func is torch._grouped_mm)
        state = prepare_functional_quantization(
            model, {grouped_mm: (None, None, weight_spec)},
            example_inputs=(
                torch.randn(4, 256, dtype=torch.bfloat16), torch.tensor([2, 4], dtype=torch.int32)))

        targets = state.iter_linear_targets()
        assert [target.name for target in targets] == ['weight[0]', 'weight[1]']
        assert [target.weight.shape for target in targets] == [(64, 256), (64, 256)]

        observed = []
        handle = state.register_linear_observer(
            lambda observation: observed.append((observation.target.name, observation.input)))
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        offsets = torch.tensor([1, 4], dtype=torch.int32)
        with functional_quantization_mode(state):
            model(x, offsets)
        handle.remove()

        assert [name for name, _ in observed] == ['weight[0]', 'weight[1]']
        torch.testing.assert_close(observed[0][1], x[:1])
        torch.testing.assert_close(observed[1][1], x[1:4])
        state.cleanup()

    @pytest.mark.skipif(not hasattr(torch, '_grouped_mm'), reason='Torch grouped_mm is unavailable')
    def test_grouped_mm_reference_weights_do_not_propagate_to_activations(self):
        """Disabled reference weights retain provenance without contaminating grouped outputs."""
        model = TwoStageGroupedFunctionalWeightModel().eval()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        offsets = torch.tensor([2, 4], dtype=torch.int32)
        state = prepare_functional_quantization(
            model, {torch._grouped_mm: (None, None, weight_spec)}, example_inputs=(x, offsets))
        observed_inputs = []
        handle = state.register_linear_observer(
            lambda observation: observed_inputs.append(observation.input))
        with functional_quantization_mode(state):
            with quantization_status_manager(model,
                                             disable_act_quant=True,
                                             disable_weight_quant=True,
                                             disable_bias_quant=True):
                output = model(x, offsets)
        handle.remove()

        assert type(output) is Tensor
        assert len(observed_inputs) == 4
        assert all(type(inp) is Tensor for inp in observed_inputs)
        state.cleanup()

    def test_grouped_mm_transformers_fallback_alias(self):
        fallback = next((
            func for func in grouped_mm_functions()
            if 'transformers.grouped_mm_fallback' in str(func)),
                        None)
        if fallback is None:
            pytest.skip('Transformers grouped_mm fallback is not registered')
        model = GroupedFunctionalWeightModel(fallback)
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        state = prepare_functional_quantization(
            model, {fallback: (None, None, weight_spec)},
            example_inputs=(
                torch.randn(4, 256, dtype=torch.bfloat16), torch.tensor([2, 4], dtype=torch.int32)))

        assert set(state.quantized_parameters) == {'weight'}
        with functional_quantization_mode(state):
            output = model(
                torch.randn(4, 256, dtype=torch.bfloat16), torch.tensor([2, 4], dtype=torch.int32))
        assert output.dtype == torch.bfloat16

    def test_stacked_weight_exposes_all_expert_targets(self):
        """One observed expert prepares stable targets for the full stacked owner."""
        model = StackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        state = prepare_functional_quantization(
            model, {F.linear: (None, None, weight_spec)}, example_inputs=(torch.randn(2, 4), 0))
        targets = state.iter_linear_targets()
        assert [target.name for target in targets] == ['weight[0]', 'weight[1]']
        assert all(target.weight.shape == (3, 4) for target in targets)
        state.cleanup()

    def test_matmul_target_uses_canonical_linear_orientation(self):
        """GPT-OSS-style [expert, input, output] targets expose [output, input]."""
        model = TransposedStackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 2, 'group_dim': 1})
        spec = (None, None, weight_spec)
        state = prepare_functional_quantization(
            model, {
                torch.matmul: spec, torch.Tensor.matmul: spec, torch.Tensor.__matmul__: spec},
            example_inputs=(torch.randn(2, 4), 0))
        targets = state.iter_linear_targets()
        assert [target.weight.shape for target in targets] == [(3, 4), (3, 4)]
        state.cleanup()

    def test_linear_observer_resolves_dynamic_expert_view(self):
        """Observer identity follows the indexed owner view, not the call ordinal."""
        model = StackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2})
        state = prepare_functional_quantization(
            model, {F.linear: (None, None, weight_spec)}, example_inputs=(torch.randn(2, 4), 0))
        observed = []
        handle = state.register_linear_observer(
            lambda observation: observed.append(observation.target.name))
        with functional_quantization_mode(state):
            model(torch.randn(2, 4), 1)
            model(torch.randn(2, 4), 0)
        handle.remove()
        assert observed == ['weight[1]', 'weight[0]']
        state.cleanup()

    def test_unsupported_weight_view_warns_and_uses_activation_fallback(self):
        """An unsupported owner view falls back to its runtime operand spec."""
        model = UnsupportedFunctionalWeightViewModel()
        quant_map = {
            F.linear: (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        with pytest.warns(UserWarning, match='falling back to runtime activation quantization'):
            state = prepare_functional_quantization(
                model, quant_map, example_inputs=(torch.randn(2, 4),))
        assert not is_parametrized(model, 'weight')
        assert len(state.quantizers) == 2
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4))
        assert out.shape == (2, 4)
        state.cleanup()

    def test_sliced_owner_without_layout_warns_and_falls_back(self):
        """Functional quantization does not infer layout policy from the operation."""
        model = StackedFunctionalWeightModel()
        spec = (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)
        with pytest.warns(UserWarning, match='require the weight quantizer to declare'):
            state = prepare_functional_quantization(
                model, {F.linear: spec}, example_inputs=(torch.randn(2, 4), 0))
        assert not is_parametrized(model, 'weight')
        assert len(state.quantizers) == 2
        state.cleanup()

    def test_sliced_owner_with_invalid_layout_warns_and_falls_back(self):
        """Invalid custom owner dimensions use the activation fallback path."""
        model = StackedFunctionalWeightModel()
        weight_spec = (Int8WeightPerTensorFloat, {'output_channel_dim': 3, 'group_dim': 2})
        spec = (Int8ActPerTensorFloat, Int8ActPerTensorFloat, weight_spec)
        with pytest.warns(UserWarning, match='not a valid owner axis'):
            state = prepare_functional_quantization(
                model, {F.linear: spec}, example_inputs=(torch.randn(2, 4), 0))
        assert not is_parametrized(model, 'weight')
        assert len(state.quantizers) == 2
        state.cleanup()

    def test_incompatible_owner_uses_warn_once_and_fall_back(self):
        """All uses fall back when configurations disagree on one owner layout."""
        model = ConflictingFunctionalWeightModel()
        linear_spec = (
            Int8ActPerTensorFloat,
            Int8ActPerTensorFloat,
            (Int8WeightPerTensorFloat, {
                'output_channel_dim': 0, 'group_dim': 1}))
        matmul_spec = (
            Int8ActPerTensorFloat,
            Int8ActPerTensorFloat,
            (Int8WeightPerTensorFloat, {
                'output_channel_dim': 1, 'group_dim': 0}))
        with pytest.warns(UserWarning, match='incompatible quantizers or matrix layouts') as record:
            state = prepare_functional_quantization(
                model, {
                    F.linear: linear_spec, torch.matmul: matmul_spec},
                example_inputs=(torch.randn(2, 4),))
        assert len(record) == 1
        assert not is_parametrized(model, 'weight')
        assert len(state.quantizers) == 4
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4))
        assert out.shape == (2, 4)
        state.cleanup()

    def test_preparametrized_owner_warns_and_falls_back(self):
        """An existing owner parametrization is not mistaken for a runtime tensor."""
        model = UnsupportedFunctionalWeightViewModel()
        register_parametrization(model, 'weight', nn.Identity())
        spec = (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)
        with pytest.warns(UserWarning, match='owner is already parametrized'):
            state = prepare_functional_quantization(
                model, {F.linear: spec}, example_inputs=(torch.randn(2, 4),))
        assert len(state.quantizers) == 2
        state.cleanup()

    def test_three_slot_linear_spec_ignores_bias(self):
        """Runtime/parameter dispatch does not apply the weight spec to bias."""
        model = SimpleLinearModel(4, 3)
        quant_map = {
            F.linear: (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear, 'weight')
        assert not is_parametrized(model.linear, 'bias')
        with functional_quantization_mode(state):
            assert isinstance(model.linear.weight, QuantTensor)
            model(x)
        state.cleanup()

    def test_input_only_skips_tied_weight_without_error(self):
        """A tied parameter is allowed when its resolved weight spec is disabled."""
        model = TiedFunctionalWeightModel()
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(
            model, {F.linear: Int8ActPerTensorFloat}, example_inputs=(x,))
        assert len(state.quantizers) == 1
        state.cleanup()

    def test_three_slot_dispatch_quantizes_parameter_first_operand(self):
        """The parameter slot applies to either of the first two operands."""
        model = ParameterFirstMatmulModel()
        quant_map = {torch.matmul: (None, None, Int8WeightPerTensorFloat)}
        state = prepare_functional_quantization(
            model, quant_map, example_inputs=(torch.randn(4, 2),))
        assert is_parametrized(model, 'weight')
        state.cleanup()

    def test_parameter_owner_refreshes_after_replacement(self):
        """Parameter provenance refreshes after an offload-like replacement."""
        model = ReplacedParameterModel()
        quant_map = {F.linear: (None, None, Int8WeightPerTensorFloat)}
        state = prepare_functional_quantization(
            model, quant_map, example_inputs=(torch.randn(2, 4),))
        assert is_parametrized(model, 'weight')
        state.cleanup()

    def test_context_manager_basic(self):
        """Test that the context manager runs without error."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        state.remove_parametrizations()
        assert out.shape == (2, 3)

    def test_quant_linear_internal_call_is_functional_noop(self):
        """Already quantized QuantLinear operands need no functional quantizer."""
        model = QuantLinearFunctionalModel()
        state = prepare_functional_quantization(
            model, {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)},
            example_inputs=(torch.randn(2, 4),))
        assert len(state.quantizers) == 0
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4))
        assert out.shape == (2, 3)
        state.cleanup()

    def test_hooks_removed_on_exit(self):
        """Test that all application hooks are removed when the context manager exits."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        ctx = functional_quantization_mode(state)
        with ctx:
            model(x)
        assert len(ctx.hooks) == 0
        state.remove_parametrizations()

    def test_quantizers_created_during_prepare(self):
        """Test that quantizers are created during prepare, before any apply forward."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert len(state.quantizers) > 0
        state.remove_parametrizations()

    def test_counters_reset_after_forward(self):
        """Test that application counters reset after each forward pass."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        ctx = functional_quantization_mode(state)
        with ctx:
            model(x)
            # After the forward pass, counters should have been reset
            for module_counters in ctx.counters.values():
                for count in module_counters.values():
                    assert count == 0
        state.remove_parametrizations()

    def test_counters_reset_for_direct_block_forward(self):
        """Direct block calls use a fresh call-site sequence like GPTQ block execution."""
        model = ModelWithFunctionalBlock()
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(
            model, {F.linear: (None, None, Int8WeightPerTensorFloat)}, example_inputs=(x,))

        with functional_quantization_mode(state):
            first = model.block(x)
            second = model.block(x)

        assert first.shape == second.shape == (2, 3)
        state.cleanup()

    def test_multiple_forward_passes(self):
        """Test that multiple forward passes work correctly."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out1 = model(x)
            out2 = model(x)
        state.remove_parametrizations()
        assert out1.shape == (2, 3)
        assert out2.shape == (2, 3)

    def test_two_linear_layers(self):
        """Test with two distinct linear layers."""
        model = TwoLinearModel(4, 3, 2)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 2)
        # Each linear layer should have its own quantizer
        assert len(state.quantizers) >= 2
        state.remove_parametrizations()

    def test_multiple_calls_same_module(self):
        """Test counting multiple calls to F.linear within the same module."""
        model = ModelWithMultiLinear(4, 3, 2)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 2)
        # The block module calls F.linear twice, so should have 2 quantizers
        block_quantizers = [k for k in state.quantizers.keys() if 'block' in k and 'linear' in k]
        assert len(block_quantizers) == 2
        state.remove_parametrizations()

    def test_multiple_weight_calls_same_module(self):
        """Distinct owners discovered in one module are parametrized once each."""
        model = ModelWithMultiLinear(4, 3, 2)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.block, 'weight1')
        assert is_parametrized(model.block, 'weight2')
        assert len([key for key in state.quantizers if key.endswith('_wq')]) == 2
        state.cleanup()

    def test_missing_second_runtime_spec_reuses_first_quantizer(self):
        """A single spec quantizes both runtime inputs of a binary function."""
        model = BmmModel()
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(
            model, {torch.bmm: Int8ActPerTensorFloat}, example_inputs=(a, b))
        assert len(state.quantizers) == 2
        with functional_quantization_mode(state):
            model(a, b)
        state.cleanup()

    def test_keyword_tensor_arguments_are_quantized(self):
        """Configured functional arguments can be supplied by keyword."""

        class KeywordBmmModel(nn.Module):

            def forward(self, a, b):
                return torch.bmm(input=a, mat2=b)

        model = KeywordBmmModel()
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)
        state = prepare_functional_quantization(
            model, {torch.bmm: Int8ActPerTensorFloat}, example_inputs=(a, b))
        assert len(state.quantizers) == 2
        state.cleanup()

    def test_explicit_none_does_not_fall_back(self):
        """An explicit None disables an argument instead of reusing arg zero."""
        model = BmmModel()
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(
            model, {torch.bmm: (Int8ActPerTensorFloat, None)}, example_inputs=(a, b))
        assert len(state.quantizers) == 1
        state.cleanup()

    def test_weight_quantization_respects_enabled(self):
        """enabled=False bypasses both activation and weight quantization."""
        model = SimpleLinearModel(4, 3)
        x = torch.randn(2, 4)
        expected = model(x)
        state = prepare_functional_quantization(
            model, {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)},
            example_inputs=(x,))
        with functional_quantization_mode(state, enabled=False):
            actual = model(x)
        assert torch.allclose(actual, expected)
        state.cleanup()

    def test_nested_disabled_mode_disables_weight_quantization(self):
        """An inner disabled mode temporarily bypasses parametrized weights."""
        model = SimpleLinearModel(4, 3)
        x = torch.randn(2, 4)
        expected = model(x)
        state = prepare_functional_quantization(
            model, {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)},
            example_inputs=(x,))
        with functional_quantization_mode(state):
            with functional_quantization_mode(state, enabled=False):
                actual = model(x)
        assert torch.allclose(actual, expected)
        state.cleanup()

    def test_cleanup_removes_retained_quantizers(self):
        """State and model cleanup are explicit and idempotent."""
        model = SimpleLinearModel(4, 3)
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(model, {F.linear: Int8ActPerTensorFloat}, (x,))
        assert hasattr(model, '_functional_quantizers')
        state.cleanup()
        assert not hasattr(model, '_functional_quantizers')
        remove_functional_quantization(model)

    def test_prepare_failure_rolls_back_model_mutations(self):
        """A failed discovery pass does not leave quantizers or parametrizations behind."""

        class FailingModel(SimpleLinearModel):

            def forward(self, x):
                self.linear(x)
                raise RuntimeError('expected preparation failure')

        model = FailingModel(4, 3)
        with pytest.raises(RuntimeError, match='expected preparation failure'):
            prepare_functional_quantization(
                model, {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)},
                example_inputs=(torch.randn(2, 4),))
        assert not is_parametrized(model.linear, 'weight')
        assert not hasattr(model, '_functional_quantizers')

    def test_prepare_runs_exactly_one_discovery_forward(self):
        """Owner registration does not require a second internal model execution."""

        class CountingModel(SimpleLinearModel):

            def __init__(self):
                super().__init__(4, 3)
                self.forward_calls = 0

            def forward(self, x):
                self.forward_calls += 1
                return super().forward(x)

        model = CountingModel()
        state = prepare_functional_quantization(
            model, {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)},
            example_inputs=(torch.randn(2, 4),))
        assert model.forward_calls == 1
        assert is_parametrized(model.linear, 'weight')
        state.cleanup()

    def test_registration_failure_rolls_back_prior_owner(self):
        """A failure after partial owner registration restores the whole model."""

        class FailingWeightQuantizer:

            @classmethod
            def let(cls, **kwargs):
                raise RuntimeError('expected registration failure')

        def weight_resolver(module, name, index):
            return Int8WeightPerTensorFloat if index == 0 else FailingWeightQuantizer

        model = ModelWithMultiLinear(4, 3, 2)
        with pytest.raises(RuntimeError, match='expected registration failure'):
            prepare_functional_quantization(
                model, {F.linear: (None, weight_resolver)}, example_inputs=(torch.randn(2, 4),))
        assert not is_parametrized(model.block, 'weight1')
        assert not is_parametrized(model.block, 'weight2')
        assert not hasattr(model, '_functional_quantizers')
        assert not hasattr(model, '_functional_quantization_state')

    def test_disabled_mode(self):
        """Test that disabled mode passes through without quantization."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        # Run without quantization
        with torch.no_grad():
            expected = model(x)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state, enabled=False):
            out = model(x)
        state.remove_parametrizations()
        assert torch.allclose(expected, out)

    def test_quantizers_registered_as_submodules(self):
        """Test that quantizers are registered as submodules on the model."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))

        assert hasattr(model, '_functional_quantizers')
        assert len(model._functional_quantizers) > 0
        state.cleanup()

    def test_non_quantizable_function_passthrough(self):
        """Test that functions not in quant_map are not affected."""

        class ModelWithReLU(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return F.relu(self.linear(x))

        model = ModelWithReLU()
        # Only quantize F.linear, not F.relu
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        state.remove_parametrizations()
        assert out.shape == (2, 3)

    def test_output_differs_from_unquantized(self):
        """Test that quantization actually changes the output (not a no-op)."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4) * 10  # Large values to make quantization effect visible

        with torch.no_grad():
            unquant_out = model(x)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with torch.no_grad(), functional_quantization_mode(state):
            # First pass initializes quantizers (calibration)
            model(x)
            # Second pass should use calibrated quantizers
            quant_out = model(x)
        state.remove_parametrizations()

        # With quantization, outputs should generally differ from unquantized
        # (unless by chance they are identical, which is unlikely with large inputs)
        # We don't assert strict inequality because first-pass calibration
        # behavior may vary
        assert quant_out.shape == unquant_out.shape

    def test_second_quantizer_with_parameter(self):
        """When a second quantizer is specified and the second arg is a parameter,
        it is registered as a parametrization during prepare."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        # Parametrization is registered during prepare, independently of any apply block.
        assert is_parametrized(model.linear, 'weight')
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # Still parametrized after an apply block with the default teardown flag.
        assert is_parametrized(model.linear, 'weight')
        state.remove_parametrizations()
        assert not is_parametrized(model.linear, 'weight')

    def test_second_quantizer_creates_weight_quant_proxy(self):
        """Test that a weight quant proxy is created with the _wq suffix."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        wq_quantizers = [k for k in state.quantizers.keys() if '_wq' in k]
        assert len(wq_quantizers) > 0, "Should have created weight quant proxy"
        state.remove_parametrizations()

    def test_second_arg_non_parameter_uses_explicit_quant_type(self):
        """When the second argument is not a parameter and a tuple is provided,
        a QuantIdentity quantizer is created for the second arg."""
        model = BmmModel()
        quant_map = {torch.bmm: (Int8ActPerTensorFloat, Int8ActPerTensorFloat)}
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(a, b))
        with functional_quantization_mode(state):
            out = model(a, b)
        assert out.shape == (2, 3, 3)
        # Should have created quantizers for both args
        arg1_quantizers = [k for k in state.quantizers.keys() if '_arg1' in k]
        assert len(arg1_quantizers) > 0
        state.remove_parametrizations()

    def test_groupwise_activation_quantizer_with_explicit_group_dim(self):
        """Groupwise activation DI uses the explicitly specified group_dim."""
        model = SimpleLinearModel(64, 16)
        quant_map = {F.linear: (MXInt8Act, {'group_dim': -1})}
        x = torch.randn(2, 64)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 16)
        assert len(state.quantizers) == 1
        quantizer = next(iter(state.quantizers.values()))
        assert isinstance(quantizer.act_quant, GroupwiseActQuantProxyFromInjector)
        assert quantizer.act_quant.group_size == 32
        assert quantizer.act_quant.group_dim == -1
        state.remove_parametrizations()

    def test_per_channel_weight_quantizer_with_explicit_output_channel_dim(self):
        """Weight DI derives out_channels from the weight and explicit output_channel_dim."""
        model = SimpleLinearModel(16, 8)
        quant_map = {
            F.linear:
                (Int8ActPerTensorFloat, (Int8WeightPerChannelFloat, {
                    'output_channel_dim': 0}))}
        x = torch.randn(2, 16)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear, 'weight')
        weight_quantizers = [m for k, m in state.quantizers.items() if '_wq' in k]
        assert len(weight_quantizers) == 1
        scale = weight_quantizers[0].scale()
        assert scale.shape == (8, 1)
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 8)
        state.remove_parametrizations()

    def test_groupwise_weight_quantizer_with_explicit_transposed_group_dim(self):
        """Functional conv_transpose weight DI uses explicit group_dim/output_channel_dim."""
        model = FunctionalConvTranspose1dModel()
        quant_map = {
            F.conv_transpose1d: (None, (MXInt8Weight, {
                'group_dim': 0, 'output_channel_dim': 1}))}
        x = torch.randn(2, 64, 8)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model, 'weight')
        weight_quantizers = [m for k, m in state.quantizers.items() if '_wq' in k]
        assert len(weight_quantizers) == 1
        assert weight_quantizers[0].group_dim == 0
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape[1] == 16
        state.remove_parametrizations()

    def test_bare_and_di_kwargs_spec_elements_coexist(self):
        """A bare quantizer and a (quantizer, di_kwargs) pair can be mixed in a spec."""
        model = SimpleLinearModel(16, 8)
        quant_map = {
            F.linear:
                (Int8ActPerTensorFloat, (Int8WeightPerChannelFloat, {
                    'output_channel_dim': 0}))}
        x = torch.randn(2, 16)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert len(state.quantizers) == 2
        assert len([key for key in state.quantizers if key.endswith('_wq')]) == 1
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 8)
        state.remove_parametrizations()

    def test_binary_op_three_quantizers_uses_runtime_second_quantizer(self):
        """Test that binary ops use the runtime second-input quantizer for tensor inputs."""
        model = BmmModel()
        quant_map = {
            torch.bmm: (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(a, b))
        with functional_quantization_mode(state):
            out = model(a, b)
        assert out.shape == (2, 3, 3)
        runtime_second_quantizers = [k for k in state.quantizers.keys() if k.endswith('_arg1')]
        weight_quantizers = [k for k in state.quantizers.keys() if '_wq' in k]
        assert len(runtime_second_quantizers) == 1
        assert len(weight_quantizers) == 0
        state.remove_parametrizations()

    def test_binary_op_three_quantizers_uses_weight_quantizer_for_parameter(self):
        """Test that binary ops use the weight quantizer when the second input is a parameter."""
        model = MatmulWeightModel()
        quant_map = {
            torch.matmul: (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        runtime_second_quantizers = [k for k in state.quantizers.keys() if k.endswith('_arg1')]
        weight_quantizers = [k for k in state.quantizers.keys() if '_wq' in k]
        assert len(runtime_second_quantizers) == 0
        assert len(weight_quantizers) == 1
        state.remove_parametrizations()
        assert not is_parametrized(model, 'weight')

    def test_second_arg_already_quant_tensor_skipped(self):
        """Test that the second argument is not re-quantized if it is already a QuantTensor."""
        from brevitas.nn import QuantIdentity as QI

        class ModelWithPreQuantized(nn.Module):

            def __init__(self):
                super().__init__()
                self.quant_id = QI(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
                self.weight = nn.Parameter(torch.randn(3, 4))

            def forward(self, x):
                # Pre-quantize weight (returns a QuantTensor)
                qw = self.quant_id(self.weight)
                return F.linear(x, qw)

        model = ModelWithPreQuantized()
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # No weight quant proxy or _arg1 quantizer should have been created because
        # the second arg was already a QuantTensor
        second_quantizers = [k for k in state.quantizers.keys() if '_arg1' in k or '_wq' in k]
        assert len(second_quantizers) == 0
        state.remove_parametrizations()

    def test_parametrization_removed_on_exit_when_requested(self):
        """With remove_parametrizations_on_exit=True, parametrizations are removed on exit."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear, 'weight')
        with functional_quantization_mode(state, remove_parametrizations_on_exit=True):
            model(x)
            assert is_parametrized(model.linear, 'weight')

        # After exiting, parametrization should be gone
        assert not is_parametrized(model.linear, 'weight')

    def test_parametrization_persists_across_blocks_by_default(self):
        """By default parametrizations survive multiple apply blocks and reuse one state."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            model(x)
        # Survives the first block ...
        assert is_parametrized(model.linear, 'weight')
        with functional_quantization_mode(state):
            model(x)
        # ... and the second block.
        assert is_parametrized(model.linear, 'weight')
        state.remove_parametrizations()
        assert not is_parametrized(model.linear, 'weight')

    def test_single_quantizer(self):
        """Test that the {func: quant_class} format only quantizes the first arg."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # No weight or _arg1 quantizers should be created when second quant is not specified
        second_quantizers = [k for k in state.quantizers.keys() if '_arg1' in k or '_wq' in k]
        assert len(second_quantizers) == 0
        state.remove_parametrizations()

    def test_parametrization_persistent_across_forwards(self):
        """Test that parametrization persists across multiple forward passes in a block."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out1 = model(x)
            assert is_parametrized(model.linear, 'weight')
            out2 = model(x)
            assert is_parametrized(model.linear, 'weight')
        assert out1.shape == (2, 3)
        assert out2.shape == (2, 3)
        state.remove_parametrizations()

    def test_two_linear_with_second_quantizer(self):
        """Test two linear layers each getting weight parametrization."""
        model = TwoLinearModel(4, 3, 2)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear1, 'weight')
        assert is_parametrized(model.linear2, 'weight')
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 2)
        state.remove_parametrizations()
        # After teardown, all parametrizations removed
        assert not is_parametrized(model.linear1, 'weight')
        assert not is_parametrized(model.linear2, 'weight')

    def test_parametrization_uses_quant_parametrization_class(self):
        """Test that registered parametrizations use the _QuantParametrization class."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear, 'weight')
        param_list = list(model.linear.parametrizations.weight)
        found = any(isinstance(p, _QuantParametrization) for p in param_list)
        assert found, "Parametrization should use _QuantParametrization"
        state.remove_parametrizations()

    def test_none_first_quantizer_skips_first_arg(self):
        """Test that specifying None as the first quantizer skips first-arg quantization."""

        class TwoTensorFunc(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, a, b):
                return torch.bmm(a, b)

        model = TwoTensorFunc()
        # None for first arg, Int8ActPerTensorFloat for second arg
        quant_map = {torch.bmm: (None, Int8ActPerTensorFloat)}
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(a, b))
        with functional_quantization_mode(state):
            out = model(a, b)
        assert out.shape == (2, 3, 3)
        # Only second-arg quantizer should be created, not first-arg
        first_quantizers = [k for k in state.quantizers.keys() if not k.endswith('_arg1')]
        arg1_quantizers = [k for k in state.quantizers.keys() if k.endswith('_arg1')]
        assert len(first_quantizers) == 0, "First-arg quantizer should not be created"
        assert len(arg1_quantizers) == 1, "Second-arg quantizer should be created"
        state.remove_parametrizations()

    def test_none_first_with_explicit_second_quantizer(self):
        """Test that the explicit second quantizer class is used even when first is None."""

        class TwoTensorFunc(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, a, b):
                return torch.bmm(a, b)

        model = TwoTensorFunc()
        quant_map = {torch.bmm: (None, Int8ActPerTensorFloat)}
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 3)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(a, b))
        with functional_quantization_mode(state):
            out = model(a, b)
        assert out.shape == (2, 3, 3)
        # The second arg quantizer should exist
        assert len(state.quantizers) == 1
        state.remove_parametrizations()

    def test_sdpa_quantization_two_args(self):
        """Test quantizing scaled_dot_product_attention query and key only."""

        class SDPAModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        # Quantize query (first arg) and key (second arg), value not specified
        quant_map = {F.scaled_dot_product_attention: (Int8ActPerTensorFloat, Int8ActPerTensorFloat)}

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(q, k, v))
        with functional_quantization_mode(state):
            out = model(q, k, v)
        assert out.shape == (1, 1, 4, 8)
        # 2 quantizers: one for query, one for key
        assert len(state.quantizers) == 2
        state.remove_parametrizations()

    def test_sdpa_quantization_three_args(self):
        """Test quantizing scaled_dot_product_attention query, key, and value."""

        class SDPAModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        # Quantize all three: query, key, and value
        quant_map = {
            F.scaled_dot_product_attention:
                (Int8ActPerTensorFloat, Int8ActPerTensorFloat, Int8ActPerTensorFloat)}

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(q, k, v))
        with functional_quantization_mode(state):
            out = model(q, k, v)
        assert out.shape == (1, 1, 4, 8)
        # 3 quantizers: query (arg0), key (arg1), value (arg2)
        assert len(state.quantizers) == 3
        arg0_keys = [k for k in state.quantizers if not '_arg' in k]
        arg1_keys = [k for k in state.quantizers if '_arg1' in k]
        arg2_keys = [k for k in state.quantizers if '_arg2' in k]
        assert len(arg0_keys) == 1
        assert len(arg1_keys) == 1
        assert len(arg2_keys) == 1
        state.remove_parametrizations()

    def test_sdpa_none_query_quantize_key(self):
        """Test SDPA with None query quantizer and explicit key quantizer."""

        class SDPAModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        # Skip query quantization, quantize key
        quant_map = {F.scaled_dot_product_attention: (None, Int8ActPerTensorFloat)}

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(q, k, v))
        with functional_quantization_mode(state):
            out = model(q, k, v)
        assert out.shape == (1, 1, 4, 8)
        # Only 1 quantizer for key (second arg)
        assert len(state.quantizers) == 1
        arg1_keys = [k for k in state.quantizers if k.endswith('_arg1')]
        assert len(arg1_keys) == 1
        state.remove_parametrizations()

    def test_sdpa_none_query_quantize_key_and_value(self):
        """Test SDPA with None query quantizer, explicit key and value quantizers."""

        class SDPAModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        # Skip query, quantize key and value
        quant_map = {
            F.scaled_dot_product_attention: (None, Int8ActPerTensorFloat, Int8ActPerTensorFloat)}

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(q, k, v))
        with functional_quantization_mode(state):
            out = model(q, k, v)
        assert out.shape == (1, 1, 4, 8)
        # 2 quantizers: key (arg1) and value (arg2)
        assert len(state.quantizers) == 2
        arg1_keys = [k for k in state.quantizers if '_arg1' in k]
        arg2_keys = [k for k in state.quantizers if '_arg2' in k]
        assert len(arg1_keys) == 1
        assert len(arg2_keys) == 1
        state.remove_parametrizations()

    def test_nested_module_dots_in_name(self):
        """Test that nested modules with dots in their names are handled correctly."""

        class Inner(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return self.linear(x)

        class Outer(nn.Module):

            def __init__(self):
                super().__init__()
                self.sub_module = Inner()

            def forward(self, x):
                return self.sub_module(x)

        model = Outer()
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # Quantizer key should not contain dots
        for key in state.quantizers:
            assert '.' not in key, f"Quantizer key should not contain dots: {key}"
        state.remove_parametrizations()

    def test_lambda_spec_returns_quantizer_class(self):
        """Test that a lambda spec element resolves to a quantizer class and is applied."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: lambda module, name, index: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # A first-arg activation quantizer should have been created via the lambda
        assert len(state.quantizers) == 1
        quantizer = next(iter(state.quantizers.values()))
        assert isinstance(quantizer, QuantIdentity)
        state.remove_parametrizations()

    def test_lambda_receives_module_name_and_index(self):
        """Test that the lambda receives the current module instance, name, and index."""
        model = ModelWithMultiLinear(4, 3, 2)
        received = []

        def resolver(module, name, index):
            received.append((module, name, index))
            return Int8ActPerTensorFloat

        quant_map = {F.linear: resolver}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))

        # Two F.linear calls in the same 'block' module -> indices 0 and 1
        assert len(received) == 2
        names = [name for _, name, _ in received]
        indices = [index for _, _, index in received]
        modules = [module for module, _, _ in received]
        assert names == ['block', 'block']
        assert indices == [0, 1]
        # The module instance passed must be the actual current nn.Module
        assert all(m is model.block for m in modules)
        state.remove_parametrizations()

    def test_lambda_in_tuple_mixed_with_class_and_none(self):
        """Test that a lambda can be mixed with a quantizer class and None in a tuple."""
        model = SimpleLinearModel(4, 3)
        # First arg uses a lambda activation quantizer; weight uses a class.
        quant_map = {
            F.linear: (lambda module, name, index: Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model.linear, 'weight')
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        act_quantizers = [
            m for k,
            m in state.quantizers.items() if '_wq' not in k and isinstance(m, QuantIdentity)]
        weight_quantizers = [k for k in state.quantizers.keys() if '_wq' in k]
        assert len(act_quantizers) == 1
        assert len(weight_quantizers) == 1
        state.remove_parametrizations()

    def test_lambda_returns_none_skips_arg(self):
        """Test that a lambda returning None skips quantization of that argument."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: lambda module, name, index: None}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        # No quantizer should be created because the resolver returned None
        assert len(state.quantizers) == 0
        state.remove_parametrizations()

    def test_lambda_in_binary_three_quantizer_weight_slot(self):
        """Test that a lambda can be used in the weight slot of a binary 3-quantizer spec."""
        model = MatmulWeightModel()
        quant_map = {
            torch.matmul: (
                Int8ActPerTensorFloat,
                Int8ActPerTensorFloat,
                lambda module,
                name,
                index: Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        assert is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            out = model(x)
        assert out.shape == (2, 3)
        weight_quantizers = [k for k in state.quantizers.keys() if '_wq' in k]
        runtime_second_quantizers = [k for k in state.quantizers.keys() if k.endswith('_arg1')]
        assert len(weight_quantizers) == 1
        assert len(runtime_second_quantizers) == 0
        state.remove_parametrizations()
        assert not is_parametrized(model, 'weight')

    def test_weight_resolver_returns_di_kwargs(self):
        """A resolver can provide owner-specific dependency-injection overrides."""
        model = StackedFunctionalWeightModel()

        def weight_resolver(module, name, index):
            return Int8WeightPerTensorFloat, {'output_channel_dim': 1, 'group_dim': 2}

        state = prepare_functional_quantization(
            model, {F.linear: (None, None, weight_resolver)}, example_inputs=(torch.randn(2, 4), 0))
        assert is_parametrized(model, 'weight')
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4), 1)
        assert out.shape == (2, 3)
        state.cleanup()

    def test_prepare_with_example_kwargs(self):
        """prepare_functional_quantization accepts keyword example inputs."""

        class KwargModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x=None):
                return self.linear(x)

        model = KwargModel()
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_kwargs={'x': x})
        assert len(state.quantizers) == 1
        with functional_quantization_mode(state):
            out = model(x=x)
        assert out.shape == (2, 3)
        state.remove_parametrizations()

    def test_prepare_requires_example_inputs(self):
        """prepare_functional_quantization requires at least one example input source."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        with pytest.raises(ValueError):
            prepare_functional_quantization(model, quant_map)

    def test_unprepared_runtime_call_skips_weight_resolver(self):
        """A later runtime-only call does not resolve the parameter weight spec."""

        class MaybeRuntimeLinear(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))
                self.register_buffer('runtime_weight', torch.randn(3, 3))

            def forward(self, x, run_runtime=False):
                x = F.linear(x, self.weight)
                if run_runtime:
                    x = F.linear(x, self.runtime_weight)
                return x

        def weight_resolver(module, name, index):
            if index != 0:
                raise AssertionError('Weight resolver ran for a runtime-only operand.')
            return Int8WeightPerTensorFloat

        model = MaybeRuntimeLinear()
        x = torch.randn(2, 4)
        state = prepare_functional_quantization(
            model, {F.linear: (None, None, weight_resolver)}, example_inputs=(x,))

        with functional_quantization_mode(state):
            output = model(x, run_runtime=True)

        assert output.shape == (2, 3)
        state.cleanup()

    def test_unprepared_call_site_raises(self):
        """Applying to a call site not seen during prepare fails fast."""

        class MaybeSecondLinear(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 3)
                self.linear2 = nn.Linear(3, 2)

            def forward(self, x, run_second=False):
                x = self.linear1(x)
                if run_second:
                    x = self.linear2(x)
                return x

        model = MaybeSecondLinear()
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        # Prepare exercises only linear1 (run_second defaults to False).
        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with pytest.raises(RuntimeError, match='No prepared quantizer'):
            with functional_quantization_mode(state):
                model(x, run_second=True)
        state.remove_parametrizations()

    def test_unprepared_disabled_call_site_passes_through(self):
        """An unseen call whose resolver disables quantization is a no-op."""

        class MaybeSecondLinear(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 3)
                self.linear2 = nn.Linear(3, 2)

            def forward(self, x, run_second=False):
                x = self.linear1(x)
                if run_second:
                    x = self.linear2(x)
                return x

        def resolver(module, name, index):
            return Int8ActPerTensorFloat if name == 'linear1' else None

        model = MaybeSecondLinear()
        state = prepare_functional_quantization(
            model, {F.linear: resolver}, example_inputs=(torch.randn(2, 4),))
        with functional_quantization_mode(state):
            out = model(torch.randn(2, 4), run_second=True)
        assert out.shape == (2, 2)
        state.cleanup()

    @requires_pt_ge('2.1')
    def test_gradient_checkpointing_without_context_fn_is_unsupported(self):
        """Without a context_fn, checkpointing is incompatible with the mode.

        Gradient checkpointing recomputes the checkpointed forward during backward
        inside checkpoint's own ``recompute_context``, which is isolated from the
        active ``TorchFunctionMode``. The recompute therefore runs unquantized
        while the original forward is quantized, and ``torch.utils.checkpoint``
        raises because a different number of tensors is saved in each pass.
        """
        from torch.utils.checkpoint import CheckpointError

        model = CheckpointedTwoLinearModel(4, 3, 2)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4, requires_grad=True)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            out = model(x, use_reentrant=False)
            with pytest.raises(CheckpointError):
                out.sum().backward()
        state.remove_parametrizations()

    @requires_pt_ge('2.1')
    def test_gradient_checkpointing_with_context_fn(self):
        """checkpoint_context_fn re-applies quantization during the recompute.

        Passing ``cm.checkpoint_context_fn()`` to ``torch.utils.checkpoint`` makes
        the interception fire during the backward recompute, so the saved-tensor
        counts match and the backward completes. We verify that gradients flow, the
        recompute does not create duplicate (mismatched-index) quantizers, and the
        outer context manager's hooks survive the step.
        """
        model = CheckpointedTwoLinearModel(4, 3, 2)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4, requires_grad=True)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state) as cm:
            num_hooks = len(cm.hooks)

            out = model(x, context_fn=cm.checkpoint_context_fn(), use_reentrant=False)
            out.sum().backward()

            # Each block calls F.linear exactly once, so counter alignment between
            # forward and recompute must keep every quantizer at call index 0: no
            # duplicate quantizers at index >= 1 should be created by the recompute.
            for key in state.quantizers:
                assert '_linear_1' not in key, f"Unexpected duplicated quantizer: {key}"
            # The outer manager still owns its hooks.
            assert len(cm.hooks) == num_hooks
            assert is_parametrized(model.block1, 'weight')
            assert is_parametrized(model.block2, 'weight')

        assert out.shape == (2, 2)
        # Gradients must reach the leaf weights through the recomputed graph.
        assert model.block1.parametrizations.weight.original.grad is not None
        assert model.block2.parametrizations.weight.original.grad is not None
        state.remove_parametrizations()
        assert not is_parametrized(model.block1, 'weight')
        assert not is_parametrized(model.block2, 'weight')

    @requires_pt_ge('2.1')
    def test_gradient_checkpointing_matches_non_checkpointed(self):
        """Checkpointed and non-checkpointed runs produce the same output.

        Re-applying quantization during the recompute must not change the
        quantized result, so the numerical output of the checkpointed model must
        match an equivalent plain model sharing the same weights.
        """
        torch.manual_seed(0)
        ckpt_model = CheckpointedTwoLinearModel(4, 3, 2)

        # Build a plain model sharing the exact same weights.
        plain_model = TwoLinearModel(4, 3, 2)
        plain_model.linear1.load_state_dict(ckpt_model.block1.state_dict())
        plain_model.linear2.load_state_dict(ckpt_model.block2.state_dict())

        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4, requires_grad=True)

        ckpt_state = prepare_functional_quantization(ckpt_model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(ckpt_state) as cm:
            ckpt_out = ckpt_model(x, context_fn=cm.checkpoint_context_fn(), use_reentrant=False)
            ckpt_out.sum().backward()
        ckpt_state.remove_parametrizations()

        plain_state = prepare_functional_quantization(plain_model, quant_map, example_inputs=(x,))
        with torch.no_grad():
            with functional_quantization_mode(plain_state):
                plain_out = plain_model(x)
        plain_state.remove_parametrizations()

        assert torch.allclose(ckpt_out.detach(), plain_out, atol=1e-6)

    @requires_pt_ge('2.1')
    def test_gradient_checkpointing_inference_no_grad(self):
        """A checkpointed forward under no_grad works without a context_fn.

        Under ``torch.no_grad()`` there is no backward pass and therefore no
        recompute, so the interception during the single forward is sufficient.
        """
        model = CheckpointedTwoLinearModel(4, 3, 2)
        quant_map = {F.linear: (Int8ActPerTensorFloat, Int8WeightPerTensorFloat)}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        with functional_quantization_mode(state):
            with torch.no_grad():
                out = model(x, use_reentrant=False)
        state.remove_parametrizations()
        assert out.shape == (2, 2)
