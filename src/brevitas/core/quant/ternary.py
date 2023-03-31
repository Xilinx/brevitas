# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant.delay import DelayWrapper
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops_ste import ternary_sign_ste


class TernaryQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that implements scaled uniform ternary quantization of an input tensor.
    Quantization is performed with :func:`~brevitas.function.ops_ste.ternary_sign_ste`.

    Args:
        scaling_impl (Module): Module that returns a scale factor.
        threshold (float): Ternarization threshold w.r.t. to the scale factor.
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale,
            zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> ternary_quant = TernaryQuant(ConstScaling(1.0), 0.5)
        >>> inp = torch.Tensor([0.04, -0.6, 3.3])
        >>> out, scale, zero_point, bit_width = ternary_quant(inp)
        >>> out
        tensor([ 0., -1.,  1.])
        >>> scale
        tensor(1.)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(2.)

    Note:
        Maps to quant_type == QuantType.TERNARY == 'TERNARY' == 'ternary' in higher-level APIs.

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    __constants__ = ['threshold']

    def __init__(self, scaling_impl: Module, threshold: float, quant_delay_steps: int = None):
        super(TernaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.threshold = threshold
        self.bit_width = BitWidthConst(2)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        mask = x.abs().gt(self.threshold * scale)
        y = mask.float() * ternary_sign_ste(x)
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()
