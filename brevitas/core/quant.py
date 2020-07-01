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

from enum import auto
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from brevitas.utils.python_utils import AutoName
from brevitas.function.ops import tensor_clamp, min_int, max_int, max_uint
from brevitas.function.ops_ste import tensor_clamp_ste, binary_sign_ste, ternary_sign_ste

from .bit_width import BitWidthConst
from .utils import StatelessBuffer


__all__ = ['QuantType', 'BinaryQuant', 'TernaryQuant', 'RescalingIntQuant',
           'PrescaledRestrictIntQuant']


class QuantType(AutoName):
    BINARY = auto()
    TERNARY = auto()
    INT = auto()
    FP = auto()


class IdentityQuant(torch.jit.ScriptModule):
    """ Placeholder Class that returns the input without performing any operation. The scale and bit_width output
    arguments are set to 0.
    """
    def __init__(self, scaling_impl: Module):
        super(IdentityQuant, self).__init__()
        self.zero = StatelessBuffer(torch.tensor(0.0))

    @torch.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return x, self.zero(), self.zero()


class BinaryQuant(torch.jit.ScriptModule):
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
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(BinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)

    @torch.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl()
        y = binary_sign_ste(x) * scale
        return y, scale, self.bit_width()


class ClampedBinaryQuant(torch.jit.ScriptModule):
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
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(ClampedBinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)

    @torch.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl()
        y = tensor_clamp(x, - scale, scale)
        y = binary_sign_ste(y) * scale
        return y, scale, self.bit_width()


class TernaryQuant(torch.jit.ScriptModule):
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
    __constants__ = ['threshold', 'bit_width']

    def __init__(self, scaling_impl: Module, threshold: float):
        super(TernaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.threshold = threshold
        self.bit_width = BitWidthConst(2)

    @torch.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl()
        mask = x.abs().ge(self.threshold * scale)
        y = mask.float() * ternary_sign_ste(x)
        y = y * scale
        return y, scale, self.bit_width()


class PrescaledRestrictIntQuantWithInputBitWidth(torch.jit.ScriptModule):
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
                 narrow_range: bool,
                 signed: bool,
                 tensor_clamp_impl: Module,
                 msb_clamp_bit_width_impl: Module,
                 float_to_int_impl: Module):
        super(PrescaledRestrictIntQuantWithInputBitWidth, self).__init__()
        self.int_quant = IntQuant(signed=signed,
                                  narrow_range=narrow_range,
                                  tensor_clamp_impl=tensor_clamp_impl,
                                  float_to_int_impl=float_to_int_impl)
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl
        self.int_scale = StatelessBuffer(torch.tensor(1.0))

    @torch.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                input_bit_width: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(input_bit_width)
        y = self.int_quant(scale, self.int_scale(), msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class PrescaledRestrictIntQuant(torch.jit.ScriptModule):
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
                 narrow_range: bool,
                 signed: bool,
                 tensor_clamp_impl: Module,
                 msb_clamp_bit_width_impl: Module,
                 float_to_int_impl: Module):
        super(PrescaledRestrictIntQuant, self).__init__()
        self.int_quant = IntQuant(signed=signed,
                                  narrow_range=narrow_range,
                                  tensor_clamp_impl=tensor_clamp_impl,
                                  float_to_int_impl=float_to_int_impl)
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl
        self.int_scale = StatelessBuffer(torch.tensor(1.0))


    @torch.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl()
        y = self.int_quant(scale, self.int_scale(), msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class IdentityPrescaledIntQuant(torch.jit.ScriptModule):
    """ Placeholder Class that returns the input without performing any operation.
    """
    @torch.jit.script_method
    def forward(self, x, input_scale, input_bit_width) -> Tuple[Tensor, Tensor, Tensor]:
        return x, input_scale, input_bit_width


class RescalingIntQuant(torch.jit.ScriptModule):
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
                 narrow_range: bool,
                 signed: bool,
                 scaling_impl: Module,
                 int_scaling_impl: Module,
                 tensor_clamp_impl: Module,
                 msb_clamp_bit_width_impl: Module,
                 float_to_int_impl: Module):
        super(RescalingIntQuant, self).__init__()
        self.int_quant = IntQuant(signed=signed,
                                  narrow_range=narrow_range,
                                  tensor_clamp_impl=tensor_clamp_impl,
                                  float_to_int_impl=float_to_int_impl)
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl

    @staticmethod
    def scaling_init_from_min_max(min_val_init: Union[int, float], max_val_init: Union[int, float]) -> torch.Tensor:
        """ Static Method that is used in the step of initializing the scale factor

        Parameters
        ----------
        min_val_init: Tensor
            Minimum value used for initialization
        max_val_init: Tensor
            Maximum value used for initialization

        Returns
        -------
        Tensor
            The largest number, in absolute value, between `max_val_init` and `min_val_init`
        """
        scaling_init = max(abs(float(min_val_init)), abs(float(max_val_init)))
        return torch.tensor(scaling_init)

    @torch.jit.script_method
    def forward(self,
                x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl()
        scale = self.scaling_impl(x)
        int_scale = self.int_scaling_impl(msb_clamp_bit_width)
        y = self.int_quant(scale, int_scale, msb_clamp_bit_width, x)
        output_bit_width = msb_clamp_bit_width
        output_scale = scale / int_scale
        return y, output_scale, output_bit_width


class IntQuant(torch.jit.ScriptModule):
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

    def __init__(self,
                 narrow_range: bool,
                 signed: bool,
                 float_to_int_impl: Module,
                 tensor_clamp_impl: Module):
        super(IntQuant, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.signed = signed
        self.narrow_range = narrow_range

    def to_int(self,
               scale: Tensor,
               int_scale: Tensor,
               msb_clamp_bit_width: Tensor,
               x: Tensor) -> Tensor:
        y = x / scale
        y = y * int_scale
        min_int_val = self.min_int(msb_clamp_bit_width)
        max_int_val = self.max_int(msb_clamp_bit_width)
        y = self.tensor_clamp_impl(y, min_val=min_int_val, max_val=max_int_val)
        y = self.float_to_int_impl(y)
        return y

    @torch.jit.script_method
    def min_int(self, bit_width):
        return min_int(self.signed, self.narrow_range, bit_width)

    @torch.jit.script_method
    def max_int(self, bit_width):
        return max_int(self.signed, bit_width)

    @torch.jit.script_method
    def max_uint(self, bit_width):
        return max_uint(self.narrow_range, bit_width)

    @torch.jit.script_method
    def forward(self,
                scale: Tensor,
                int_scale: Tensor,
                msb_clamp_bit_width: Tensor,
                x: Tensor) -> Tensor:
        y_int = self.to_int(scale, int_scale, msb_clamp_bit_width, x)
        y = y_int / int_scale
        y = y * scale
        return y
