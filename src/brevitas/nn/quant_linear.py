# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
from typing import Optional
from typing import Type
from typing import Union
import weakref

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import Linear
from torch.nn.functional import linear
from torch.utils.checkpoint import checkpoint

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.quant_tensor import QuantTensor

from .quant_layer import ActQuantType
from .quant_layer import BiasQuantType
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from .quant_layer import WeightQuantType

__all__ = ['QuantLinear']


class _RecomputeQuantLinearFn(Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, module, *quant_params):
        ctx.module_ref = weakref.ref(module)
        ctx.has_bias = bias is not None
        ctx.autocast_enabled = torch.is_autocast_enabled()
        ctx.autocast_dtype = torch.get_autocast_gpu_dtype() if inp.is_cuda else None
        ctx.save_for_backward(inp)
        quant_weight = module.quant_weight()
        return linear(inp, quant_weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        module = ctx.module_ref()
        (saved_input,) = ctx.saved_tensors
        autocast = (
            torch.autocast('cuda', enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype)
            if saved_input.is_cuda else nullcontext())
        with torch.enable_grad(), autocast:
            inp = saved_input.detach().requires_grad_(ctx.needs_input_grad[0])
            weight = module.weight
            bias = module.bias if ctx.has_bias else None
            quant_params = tuple(module.weight_quant.parameters())
            quant_weight = module.quant_weight()
            output = linear(inp, quant_weight, bias)

            targets = [inp, weight, bias, *quant_params]
            active_indices = [
                index for index,
                target in enumerate(targets) if target is not None and target.requires_grad]
            active_grads = torch.autograd.grad(
                output, [targets[index] for index in active_indices],
                grad_output,
                allow_unused=True)

        grads = [None] * len(targets)
        for index, grad in zip(active_indices, active_grads):
            grads[index] = grad
        grad_input, grad_weight, grad_bias, *grad_quant_params = grads
        return grad_input, grad_weight, grad_bias, None, *grad_quant_params


class QuantLinear(QuantWBIOL, Linear):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: Optional[bool] = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            quant_checkpointing: bool = False,
            quant_recompute: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs) -> None:
        Linear.__init__(self, in_features, out_features, bias, device=device, dtype=dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.quant_checkpointing = quant_checkpointing
        self.quant_recompute = quant_recompute

    @property
    def per_elem_ops(self):
        return 2 * self.in_features

    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.out_features

    @property
    def channelwise_separable(self) -> bool:
        return False

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        if (self.quant_checkpointing and self.training and torch.is_grad_enabled() and
                not self.export_mode and not torch.jit.is_scripting() and
                not torch.jit.is_tracing()):
            return checkpoint(self._forward_impl, input, use_reentrant=False)
        return self._forward_impl(input)

    def _forward_impl(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        if (self.quant_recompute and self.training and torch.is_grad_enabled() and
                not self.export_mode and not torch.jit.is_scripting() and
                not torch.jit.is_tracing() and not self.input_quant.is_quant_enabled and
                not self.bias_quant.is_quant_enabled and not self.output_quant.is_quant_enabled and
                not self.return_quant_tensor and isinstance(input, Tensor)):
            return _RecomputeQuantLinearFn.apply(
                input, self.weight, self.bias, self, *tuple(self.weight_quant.parameters()))
        return self.forward_impl(input)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        output_tensor = linear(x, quant_weight, quant_bias)
        return output_tensor

    def quant_output_scale_impl(
            self, inp: Tensor, quant_input_scale: Tensor, quant_weight_scale: Tensor):
        if quant_input_scale.shape == ():
            input_broadcast_shape = tuple([1] * len(inp.size()))
            quant_input_scale = quant_input_scale.view(input_broadcast_shape)
        if quant_weight_scale.shape == ():
            weight_broadcast_shape = tuple([1] * len(self.weight.size()))
            quant_weight_scale = quant_weight_scale.view(weight_broadcast_shape)
        quant_output_scale = linear(quant_input_scale, quant_weight_scale)
        return quant_output_scale
