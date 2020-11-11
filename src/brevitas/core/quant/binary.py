# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops import tensor_clamp
from brevitas.function.ops_ste import binary_sign_ste
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.utils import StatelessBuffer
from .delay import DelayWrapper


class BinaryQuant(brevitas.jit.ScriptModule):
    """ Class that implement the binary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    The scale factor is determined internally through the scaling_impl module.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor

    Attributes
    ----------
    scaling_impl: Module
       Module that determines the value of the scale factor
    bit_width: Int
        For binary quantization, the bit_width is constant and fixed to 1

    Methods
    -------
    forward(x)
        Perform the binary quantization using :func:`~brevitas.function.ops_ste.binary_sign_ste`. After that, the
        result is converted to floating point through the scale factor.
        The scale factor is determined by the attribute `scaling_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.

    """

    def __init__(self, scaling_impl: Module, quant_delay_steps: int = None):
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
    """ Class that implement the binary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    Before performing the binarization, the input tensor is clamped in the range of admissible values, determined by the
    scale factor.
    The scale factor is determined internally through the scaling_impl module.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor

    Attributes
    ----------
    scaling_impl : Module
       Module that determines the value of the scale factor
    bit_width : Int
        For binary quantization, the bit_width is constant and fixed to 1

    Methods
    -------
    forward(x)
        Perform the binary quantization using :func:`~brevitas.function.ops_ste.binary_sign_ste`. After that, the
        result is converted to floating point through the scale factor.
        The scale factor is determined by the attribute `scaling_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.

    """

    def __init__(self, scaling_impl: Module, quant_delay_steps: int = None):
        super(ClampedBinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        y = tensor_clamp(x, - scale, scale)
        y = binary_sign_ste(y) * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()