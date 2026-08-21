# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import torch
from torch.utils.checkpoint import checkpoint

from brevitas.nn import QuantLinear
from brevitas.quant import Int32Bias
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant_tensor import IntQuantTensor

OUTPUT_FEATURES = 10
INPUT_FEATURES = 5
BIT_WIDTH = 5


class TestQuantLinearInit:

    def test_module_init_defaults(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=False)
        assert mod

    def test_module_init_bias_fp(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        assert mod

    def test_module_init_bias_int(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            bias_quant=Int32Bias)
        assert mod

    def test_module_init_scale_impl_type_override(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            weight_scaling_impl_type='HE')
        assert mod.weight_quant.scale()


class TestQuantLinearFwd:

    def test_forward_defaults(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        x = torch.rand(size=(3, INPUT_FEATURES))
        assert mod(x) is not None

    def test_forward_bias_fp(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        x = torch.rand(size=(3, INPUT_FEATURES))
        assert mod(x) is not None

    def test_forward_bias_int(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            bias_quant=Int32Bias)
        x = IntQuantTensor(
            torch.rand(size=(3, INPUT_FEATURES)),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(3),
            signed=True,
            training=False)
        assert mod(x) is not None


class TestQuantLinearCheckpointing:

    @staticmethod
    def checkpointed_pair():
        reference = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            weight_quant=Int8WeightPerChannelFloat,
            weight_scaling_impl_type='parameter_from_stats')
        reference.eval()
        reference(torch.randn(3, INPUT_FEATURES))
        candidate = deepcopy(reference)
        candidate.quant_checkpointing = True
        reference.train()
        candidate.train()
        return reference, candidate

    @staticmethod
    def forward_backward(module, inp, grad, outer_checkpoint=False):
        inp = inp.detach().clone().requires_grad_(True)
        if outer_checkpoint:
            output = checkpoint(module, inp, use_reentrant=False)
        else:
            output = module(inp)
        output.backward(grad)
        grads = {
            name: parameter.grad.detach().clone() for name,
            parameter in module.named_parameters() if parameter.grad is not None}
        return output.detach(), inp.grad.detach(), grads

    def test_gradient_parity(self):
        reference, candidate = self.checkpointed_pair()
        inp = torch.randn(3, INPUT_FEATURES)
        grad = torch.randn(3, OUTPUT_FEATURES)

        reference_output, reference_input_grad, reference_grads = self.forward_backward(
            reference, inp, grad)
        candidate_output, candidate_input_grad, candidate_grads = self.forward_backward(
            candidate, inp, grad)

        torch.testing.assert_close(candidate_output, reference_output)
        torch.testing.assert_close(candidate_input_grad, reference_input_grad)
        assert candidate_grads.keys() == reference_grads.keys()
        for name in reference_grads:
            torch.testing.assert_close(candidate_grads[name], reference_grads[name])

    def test_nested_checkpoint_gradient_parity(self):
        reference, candidate = self.checkpointed_pair()
        inp = torch.randn(3, INPUT_FEATURES)
        grad = torch.randn(3, OUTPUT_FEATURES)

        reference_output, reference_input_grad, reference_grads = self.forward_backward(
            reference, inp, grad, outer_checkpoint=True)
        candidate_output, candidate_input_grad, candidate_grads = self.forward_backward(
            candidate, inp, grad, outer_checkpoint=True)

        torch.testing.assert_close(candidate_output, reference_output)
        torch.testing.assert_close(candidate_input_grad, reference_input_grad)
        assert candidate_grads.keys() == reference_grads.keys()
        for name in reference_grads:
            torch.testing.assert_close(candidate_grads[name], reference_grads[name])

    def test_reduces_saved_tensor_bytes(self):
        reference, candidate = self.checkpointed_pair()
        inp = torch.randn(32, INPUT_FEATURES, requires_grad=True)

        def saved_bytes(module):
            total = 0

            def pack(tensor):
                nonlocal total
                total += tensor.numel() * tensor.element_size()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                module(inp)
            return total

        assert saved_bytes(candidate) < saved_bytes(reference)
