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

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops_ste import round_ste
from brevitas.core.utils import StatelessBuffer
from brevitas.core.quant.delay import DelayWrapper


class PrescaledRestrictIntQuantWithInputBitWidth(brevitas.jit.ScriptModule):
    """
    ScriptModule that wraps around an integer quantization implementation like
    :class:`~brevitas.core.quant.IntQuant`. Zero-point is set to zero, scale is taken as input,
    bit-width is computed from an input bit-width.

     Args:
        int_quant (Module): Module that implements integer quantization.
        bit_width_impl (Module): Module that takes the input bit-width in and returns the bit-width
            to be used for quantization.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale,
            zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> from brevitas.core.function_wrapper import Identity
        >>> from brevitas.core.quant import IntQuant
        >>> int_quant = IntQuant(narrow_range=True, signed=True)
        >>> int_quant_wrapper = PrescaledRestrictIntQuantWithInputBitWidth(int_quant, Identity())
        >>> scale, input_bit_width = torch.tensor(0.01), torch.tensor(4.)
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out, scale, zero_point, bit_width = int_quant_wrapper(inp, scale, input_bit_width)
        >>> out
        tensor([ 0.0400, -0.0500,  0.0700, -0.0700])
        >>> scale
        tensor(0.0100)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(4.)

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """
    def __init__(self,
                 int_quant: Module,
                 bit_width_impl: Module):
        super(PrescaledRestrictIntQuantWithInputBitWidth, self).__init__()
        self.int_quant = int_quant
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.zero_point = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                input_bit_width: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bit_width = self.msb_clamp_bit_width_impl(input_bit_width)
        zero_point = self.zero_point()
        y = self.int_quant(scale, zero_point, bit_width, x)
        return y, scale, zero_point, bit_width


class PrescaledRestrictIntQuant(brevitas.jit.ScriptModule):
    """
    """
    def __init__(self,
                 int_quant: Module,
                 bit_width_impl: Module):
        super(PrescaledRestrictIntQuant, self).__init__()
        self.int_quant = int_quant
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.zero_point = StatelessBuffer(torch.tensor(0.0))


    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl()
        zero_point = self.zero_point()
        y = self.int_quant(scale, zero_point, msb_clamp_bit_width, x)
        return y, scale, zero_point, msb_clamp_bit_width


class RescalingIntQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that wraps around an integer quantization implementation like
    :class:`~brevitas.core.quant.IntQuant`. Scale, zero-point and bit-width are returned from their
    respective implementations and passed on to the integer quantization implementation.

     Args:
        int_quant (Module): Module that implements integer quantization.
        scaling_impl (Module): Module that takes in the input to quantize and returns a scale factor,
            here interpreted as threshold on the floating-point range of quantization.
        int_scaling_impl (Module): Module that takes in a bit-width and returns an integer scale
            factor, here interpreted as threshold on the integer range of quantization.
        zero_point_impl (Module): Module that returns an integer zero-point.
        bit_width_impl (Module): Module that returns a bit-width.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale,
            zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> from brevitas.core.zero_point import ZeroZeroPoint
        >>> from brevitas.core.scaling import IntScaling
        >>> from brevitas.core.quant import IntQuant
        >>> from brevitas.core.bit_width import BitWidthConst
        >>> int_quant_wrapper = RescalingIntQuant(
        ...                         IntQuant(narrow_range=True, signed=True),
        ...                         ConstScaling(0.1),
        ...                         IntScaling(signed=True, narrow_range=True),
        ...                         ZeroZeroPoint(),
        ...                         BitWidthConst(4))
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out, scale, zero_point, bit_width = int_quant_wrapper(inp)
        >>> out
        tensor([ 0.0429, -0.0571,  0.1000, -0.1000])
        >>> scale
        tensor(0.0143)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(4.)

    Note:
        scale = scaling_impl(x) / int_scaling_impl(bit_width)

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """
    def __init__(self,
                 int_quant: Module,
                 scaling_impl: Module,
                 int_scaling_impl: Module,
                 zero_point_impl: Module,
                 bit_width_impl: Module):
        super(RescalingIntQuant, self).__init__()
        self.int_quant = int_quant
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.zero_point_impl = zero_point_impl
        self.msb_clamp_bit_width_impl = bit_width_impl

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bit_width = self.msb_clamp_bit_width_impl()
        threshold = self.scaling_impl(x)
        int_threshold = self.int_scaling_impl(bit_width)
        scale = threshold / int_threshold
        zero_point = self.zero_point_impl(x, scale, bit_width)
        y = self.int_quant(scale, zero_point, bit_width, x)
        return y, scale, zero_point, bit_width


class DecoupledRescalingIntQuant(brevitas.jit.ScriptModule):

    def __init__(self,
                 decoupled_int_quant: Module,
                 pre_scaling_impl: Module,
                 scaling_impl: Module,
                 int_scaling_impl: Module,
                 pre_zero_point_impl: Module,
                 zero_point_impl: Module,
                 bit_width_impl: Module):
        super(DecoupledRescalingIntQuant, self).__init__()
        self.decoupled_int_quant = decoupled_int_quant
        self.pre_scaling_impl = pre_scaling_impl
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.pre_zero_point_impl = pre_zero_point_impl
        self.zero_point_impl = zero_point_impl
        self.msb_clamp_bit_width_impl = bit_width_impl

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bit_width = self.msb_clamp_bit_width_impl()
        int_threshold = self.int_scaling_impl(bit_width)
        pre_threshold = self.pre_scaling_impl(x)
        pre_scale = pre_threshold / int_threshold
        pre_zero_point = self.pre_zero_point_impl(x, pre_scale, bit_width)
        threshold = self.scaling_impl(x)
        scale = threshold / int_threshold
        zero_point = self.zero_point_impl(x, scale, bit_width)
        y = self.decoupled_int_quant(pre_scale, pre_zero_point, scale, zero_point, bit_width, x)
        return y, pre_scale, pre_zero_point, scale, zero_point, bit_width


class TruncIntQuant(brevitas.jit.ScriptModule):
    """
    """
    def __init__(
            self,
            float_to_int_impl: Module,
            bit_width_impl: Module,
            quant_delay_steps: int = 0):
        super(TruncIntQuant, self).__init__()
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.float_to_int_impl = float_to_int_impl
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                zero_point: Tensor,
                input_bit_width: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        y = x / scale
        y = y + zero_point
        y = round_ste(y)  # clean up floating point error
        output_bit_width = self.msb_clamp_bit_width_impl()
        trunc_bit_width = input_bit_width - output_bit_width
        trunc_scale = 2.0 ** trunc_bit_width
        y = y / trunc_scale
        y = self.float_to_int_impl(y)
        y = y - zero_point
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y, scale, zero_point, output_bit_width