# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.delay import DelayWrapper
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops_ste import binary_sign_ste


class BinaryQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that implements scaled uniform binary quantization of an input tensor.
    Quantization is performed with :func:`~brevitas.function.ops_ste.binary_sign_ste`.

    Args:
        scaling_impl (Module): Module that returns a scale factor.
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale, zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> binary_quant = BinaryQuant(ConstScaling(0.1))
        >>> inp = torch.Tensor([0.04, -0.6, 3.3])
        >>> out, scale, zero_point, bit_width = binary_quant(inp)
        >>> out
        tensor([ 0.1000, -0.1000,  0.1000])
        >>> scale
        tensor(0.1000)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(1.)

    Note:
        Maps to quant_type == QuantType.BINARY == 'BINARY' == 'binary' when applied to weights in higher-level APIs.

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    def __init__(self, scaling_impl: Module, quant_delay_steps: int = 0):
        super(BinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        y = binary_sign_ste(x) * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()


class ClampedBinaryQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that implements scaled uniform binary quantization of an input tensor. Before
    going through quantization, the input tensor is clamped between (- scale, scale), which
    on the backward pass zeroes gradients corresponding to inputs outside that range.
    Quantization is performed with :func:`~brevitas.function.ops_ste.binary_sign_ste`.

    Args:
        scaling_impl (Module): Module that returns a scale factor.
        tensor_clamp_impl (Module): Module that performs tensor-wise clamping. Default TensorClamp()
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale, zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> binary_quant = ClampedBinaryQuant(ConstScaling(0.1))
        >>> inp = torch.Tensor([0.04, -0.6, 3.3]).requires_grad_(True)
        >>> out, scale, zero_point, bit_width = binary_quant(inp)
        >>> out
        tensor([ 0.1000, -0.1000,  0.1000], grad_fn=<MulBackward0>)
        >>> out.backward(torch.Tensor([1.0, 1.0, 1.0]))
        >>> inp.grad
        tensor([0.1000, 0.0000, 0.0000])
        >>> scale
        tensor(0.1000)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(1.)

    Note:
        Maps to quant_type == QuantType.BINARY == 'BINARY' == 'binary' when applied to activations
         in higher-level APIs.

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    def __init__(
            self,
            scaling_impl: Module,
            tensor_clamp_impl: Module = TensorClamp(),
            quant_delay_steps: int = 0):
        super(ClampedBinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)
        self.tensor_clamp_impl = tensor_clamp_impl

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        y = self.tensor_clamp_impl(x, -scale, scale)
        y = binary_sign_ste(y) * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()
