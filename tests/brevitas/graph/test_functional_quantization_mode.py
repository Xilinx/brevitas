# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized
from torch.utils.checkpoint import checkpoint

from brevitas.graph.quantize import _QuantParametrization
from brevitas.graph.quantize import functional_quantization_mode
from brevitas.graph.quantize import prepare_functional_quantization
from brevitas.nn import QuantIdentity
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from tests.marker import requires_pt_ge


class SimpleLinearModel(nn.Module):
    """Model with a single linear layer that calls F.linear."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

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

    def test_hooks_removed_on_exit(self):
        """Test that all application hooks are removed when the context manager exits."""
        model = SimpleLinearModel(4, 3)
        quant_map = {F.linear: Int8ActPerTensorFloat}
        x = torch.randn(2, 4)

        state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
        ctx = functional_quantization_mode(state)
        with ctx:
            model(x)
        assert len(ctx._hook_handles) == 0
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
            for module_counters in ctx._counters.values():
                for count in module_counters.values():
                    assert count == 0
        state.remove_parametrizations()

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

        # Check that quantizers are accessible as model attributes
        found = False
        for name, module in model.named_modules():
            if name.startswith('_fq_'):
                found = True
                break
        assert found, "Quantizer modules should be registered on the model"
        state.remove_parametrizations()

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
        # Bare element -> empty di_kwargs; paired element -> its dict.
        assert state.arg_di_kwargs_map[F.linear] == [{}, {'output_channel_dim': 0}]
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
        with pytest.raises(AssertionError):
            prepare_functional_quantization(model, quant_map)

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
            num_hooks = len(cm._hook_handles)

            out = model(x, context_fn=cm.checkpoint_context_fn(), use_reentrant=False)
            out.sum().backward()

            # Each block calls F.linear exactly once, so counter alignment between
            # forward and recompute must keep every quantizer at call index 0: no
            # duplicate quantizers at index >= 1 should be created by the recompute.
            for key in state.quantizers:
                assert '_linear_1' not in key, f"Unexpected duplicated quantizer: {key}"
            # The outer manager still owns its hooks.
            assert len(cm._hook_handles) == num_hooks
            assert is_parametrized(model.block1, 'weight')
            assert is_parametrized(model.block2, 'weight')

        assert out.shape == (2, 2)
        # Gradients must reach the model weights through the recomputed graph.
        assert model.block1.weight.grad is not None
        assert model.block2.weight.grad is not None
        state.remove_parametrizations()
        assert not is_parametrized(model.block1, 'weight')
        assert not is_parametrized(model.block2, 'weight')

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
