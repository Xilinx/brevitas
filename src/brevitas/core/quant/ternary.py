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
from brevitas.function.ops_ste import ternary_sign_ste
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.utils import StatelessBuffer
from .delay import DelayWrapper


class TernaryQuant(brevitas.jit.ScriptModule):
    """ Class that implement the ternary quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor.

    The scale factor is determined internally through the scaling_impl module. The threshold is a user-defined value in
    the range (0,1).

    The quantization is performed in such a way that all input values in the range
    (-scale*threshold, scale*threshold) are quantized to 0. Values greater than the upper bound are quantized to 'scale'
    . Values lower than the lower bound are quantized to '-scale'.

    Parameters
    ----------
    scaling_impl : Module
        Module that determines the value of the scale factor
    threshold: Float
        User-defined value that determines, together with the scale factor, the range of values that are quantized to 0.

    Attributes
    ----------
    scaling_impl : Module
       Module that determines the value of the scale factor
    bit_width : Int
        For binary quantization, the bit_width is constant and fixed to 2
    threshold: Float
        User-defined value that determines, together with the scale factor, the range of values that are quantized to 0.

    Methods
    -------
    forward(x)
        Perform the ternary quantization using :func:`~brevitas.function.ops_ste.ternary_sign_ste`. After that, the
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
    __constants__ = ['threshold']

    def __init__(self, scaling_impl: Module, threshold: float, quant_delay_steps: int = None):
        super(TernaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.threshold = threshold
        self.bit_width = BitWidthConst(2)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        mask = x.abs().ge(self.threshold * scale)
        y = mask.float() * ternary_sign_ste(x)
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()