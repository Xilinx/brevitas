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

from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops import max_uint, max_int, min_int
from .delay import DelayWrapper


class IntQuant(brevitas.jit.ScriptModule):
    """ Class that implement the quantization of the input tensor, which is then converted to its floating point
    representation according to the scale factor (i.e. scale/int_scale).

    All values required for the quantization are determined externally.


    Parameters
    ----------
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.

    Methods
    -------
    to_int(scale, int_scale_msb_clamp_bit_width, x)
        Perform the conversion to integer of the input tensor.
        After diving by the scale factor (i.e. scale/int_scale), the input tensor is clamped in the range of admissible
        integer values, and then converted to integer according to the strategy defined by `float_to_int_impl`.

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Floating point component of the scale factor
        int_scale: Tensor
            Integer component of the scale factor
        msb_clamp_bit_width: Tensor
            Bit_width to be used for the conversion to integer

    forward(scale, int_scale, msb_clamp_bit_width, x)
        Perform the quantization of the input tensor. The value is first converted to its integer representation and
        quantized, then converted to its floating representation multiplying it by the scale factor
        (i.e. scale/scale_int)

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Floating point component of the scale factor
        int_scale: Tensor
            Integer component of the scale factor
        msb_clamp_bit_width: Tensor
            Bit_width to be used for the conversion to integer

        Returns
        -------
        Tensor
            The quantized tensor after its conversion to floating point

    min_int(bit_width)
        Determines the minimum integer representable according to the values of `signed`, `narrow_range`, and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the minimum integer representable

        Returns
        -------
        Tensor
            The minimum integer representable

    max_int(bit_width)
        Determines the maximum signed integer representable according to the values of `signed`, `narrow_range`, and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the maximum integer representable

        Returns
        -------
        Tensor
            The maximum integer representable

    max_uint(bit_width)
        Determines the maximum unsigned integer representable according to the values of `narrow_range` and
        `bit_width`.

        Parameters
        ----------
        bit_width: Tensor
            Number of bits for determining the maximum integer representable

        Returns
        -------
        Tensor
            The maximum integer representable
    """
    __constants__ = ['signed', 'narrow_range']

    def __init__(
            self,
            narrow_range: bool,
            signed: bool,
            float_to_int_impl: Module,
            tensor_clamp_impl: Module,
            quant_delay_steps: int = None):
        super(IntQuant, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.signed = signed
        self.narrow_range = narrow_range
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method_110_disabled
    def to_int(self,
               scale: Tensor,
               int_scale: Tensor,
               zero_point: Tensor,
               msb_clamp_bit_width: Tensor,
               x: Tensor) -> Tensor:
        y = x / scale
        y = y * int_scale
        y + zero_point
        min_int_val = self.min_int(msb_clamp_bit_width)
        max_int_val = self.max_int(msb_clamp_bit_width)
        y = self.tensor_clamp_impl(y, min_val=min_int_val, max_val=max_int_val)
        y = self.float_to_int_impl(y)
        return y

    @brevitas.jit.script_method
    def min_int(self, bit_width):
        return min_int(self.signed, self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def max_int(self, bit_width):
        return max_int(self.signed, bit_width)

    @brevitas.jit.script_method
    def max_uint(self, bit_width):
        return max_uint(self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def forward(self,
                scale: Tensor,
                int_scale: Tensor,
                zero_point: Tensor,
                msb_clamp_bit_width: Tensor,
                x: Tensor) -> Tensor:
        y_int = self.to_int(scale, int_scale, zero_point, msb_clamp_bit_width, x)
        y = y_int - zero_point
        y = y / int_scale
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y