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
from .delay import DelayWrapper


class PrescaledRestrictIntQuantWithInputBitWidth(brevitas.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width.
    Scale is determined externally, int_scale is set to 1, while bit_width is determined internally through
    msb_clamp_bit_width_impl.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant : Module
       Module that performs the actual quantization
    msb_clamp_bit_width_impl : Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x, scale, input_bit_width)
        After determining internally the bit_width value, it calls IntQuant to perform the quantization of the input

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Scale factor that regulates the conversion between integer and floating point version of the input tensor
        input_bit_width
            Bit_width that, going in `msb_clamp_bit_with`, is used to determine the bit_width for the quantization

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """
    def __init__(self,
                 int_quant: Module,
                 bit_width_impl: Module):
        super(PrescaledRestrictIntQuantWithInputBitWidth, self).__init__()
        self.int_quant = int_quant
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.int_scale = StatelessBuffer(torch.tensor(1.0))
        self.zero_point = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                input_bit_width: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(input_bit_width)
        zero_point = self.zero_point()
        y = self.int_quant(scale, self.int_scale(), zero_point, msb_clamp_bit_width, x)
        return y, scale, zero_point, msb_clamp_bit_width


class PrescaledRestrictIntQuant(brevitas.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width.
    Scale is determined externally, int_scale is set to 1, while bit_width is determined internally through
    msb_clamp_bit_width_impl.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant: Module
       Module that performs the actual quantization
    msb_clamp_bit_width_impl: Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x, scale)
        After determining internally the bit_width value, it calls IntQuant to perform the quantization of the input

        Parameters
        ----------
        x: Tensor
            Input tensor that will be quantized
        scale: Tensor
            Scale factor that regulates the conversion between integer and floating point version of the input tensor

        Returns
        -------
        Tuple(Tensor, Tensor, Tensor)
            Tuple with three values where:
            y is the quantized Tensor;
            scale is the scale factor;
            bit_width is the bit_width of the quantization.
    """
    def __init__(self,
                 int_quant: Module,
                 bit_width_impl: Module):
        super(PrescaledRestrictIntQuant, self).__init__()
        self.int_quant = int_quant
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.int_scale = StatelessBuffer(torch.tensor(1.0))
        self.zero_point = StatelessBuffer(torch.tensor(0.0))


    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl()
        zero_point = self.zero_point()
        y = self.int_quant(scale, self.int_scale(), zero_point, msb_clamp_bit_width, x)
        return y, scale, zero_point, msb_clamp_bit_width


class RescalingIntQuant(brevitas.jit.ScriptModule):
    """ Wrapper around :class:`~brevitas.core.quant.IntQuant`, that is responsible for the actual quantization of the
    input.

    The modules tensor_clamp_impl and float_to_int_impl, and the booleans `signed` and `narrow_range` are required by
    `IntQuant` to perform the quantization.

    The `int_scaling_impl` module is required to  determine int_scale.

    In order to perform the actual quantization, it is required to determine the following values: scale, int_scale,
    bit_width. All values are determined internally.
    Must be noted that there is a name overload and that the actual scale factor is obtained computing scale/int_scale.

    Parameters
    ----------
    narrow_range: Bool
        Bool that determines whether to enable or not the narrow range representation.
    signed: Bool
        Bool that determines whether to use signed or unsigned integers.
    tensor_clamp_impl: Module
        Module that performs the clamping of the input values for a proper integer representation
    msb_clamp_bit_width_impl: Module
        Module that determines the bit_width for the integer conversion
    float_to_int_impl: Module
        Module that performs the conversion from floating point to integer representation

    Attributes
    ----------
    int_quant: Module
       Module that performs the actual quantization
    scaling_impl: Module
        Module that is responsible for the computation of the scale factor
    int_scaling_impl: Module
        Module that is responsible for the computation of the int_scale factor
    msb_clamp_bit_width_impl: Int
        Module that determines the bit_width for the integer conversion

    Methods
    -------
    forward(x)
        After determining internally the bit_width value, the scale factor, and the int_scale factor
        the method calls IntQuant to perform the quantization of the input.

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
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl()
        scale = self.scaling_impl(x)
        int_scale = self.int_scaling_impl(msb_clamp_bit_width)
        zero_point = self.zero_point_impl(x, scale, msb_clamp_bit_width)
        y = self.int_quant(scale, int_scale, zero_point, msb_clamp_bit_width, x)
        output_scale = scale / int_scale
        output_zero_point = zero_point
        output_bit_width = msb_clamp_bit_width
        return y, output_scale, output_zero_point, output_bit_width


class TruncIntQuant(brevitas.jit.ScriptModule):
    def __init__(
            self,
            float_to_int_impl: Module,
            bit_width_impl: Module,
            quant_delay_steps: Optional[int] = None):
        super(TruncIntQuant, self).__init__()
        self.msb_clamp_bit_width_impl = bit_width_impl
        self.float_to_int_impl = float_to_int_impl
        self.zero_point_impl = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                input_bit_width: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        y = x / scale
        y = round_ste(y)  # clean up floating point error
        output_bit_width = self.msb_clamp_bit_width_impl()
        trunc_bit_width = input_bit_width - output_bit_width
        trunc_scale = 2.0 ** trunc_bit_width
        y = y / trunc_scale
        y = self.float_to_int_impl(y)
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point_impl(), output_bit_width