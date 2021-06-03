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

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops import max_int, min_int
from brevitas.core.function_wrapper import RoundSte, TensorClamp
from brevitas.core.quant.delay import DelayWrapper


class IntQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that implements scale, shifted, uniform integer quantization of an input tensor,
    according to an input scale, zero-point and bit-width.

    Args:
        narrow_range (bool): Flag that determines whether restrict quantization to a narrow range or not.
        signed (bool): Flag that determines whether to quantize to a signed range or not.
        float_to_int_impl (Module): Module that performs the conversion from floating point to
            integer representation. Default: RoundSte()
        tensor_clamp_impl (Module): Module that performs clamping. Default: TensorClamp()
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tensor: Quantized output in de-quantized format.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> int_quant = IntQuant(narrow_range=True, signed=True)
        >>> scale, zero_point, bit_width = torch.tensor(0.01), torch.tensor(0.), torch.tensor(4.)
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out = int_quant(scale, zero_point, bit_width, inp)
        >>> out
        tensor([ 0.0400, -0.0500,  0.0700, -0.0700])

    Note:
        Maps to quant_type == QuantType.INT == 'INT' == 'int' in higher-level APIs.

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    __constants__ = ['signed', 'narrow_range']

    def __init__(
            self,
            narrow_range: bool,
            signed: bool,
            float_to_int_impl: Module = RoundSte(),
            tensor_clamp_impl: Module = TensorClamp(),
            quant_delay_steps: int = 0):
        super(IntQuant, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.signed = signed
        self.narrow_range = narrow_range
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method_110_disabled
    def to_int(
            self,
            scale: Tensor,
            zero_point: Tensor,
            bit_width: Tensor,
            x: Tensor) -> Tensor:
        y = x / scale
        y = y + zero_point
        min_int_val = self.min_int(bit_width)
        max_int_val = self.max_int(bit_width)
        y = self.tensor_clamp_impl(y, min_val=min_int_val, max_val=max_int_val)
        y = self.float_to_int_impl(y)
        return y

    @brevitas.jit.script_method
    def min_int(self, bit_width):
        return min_int(self.signed, self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def max_int(self, bit_width):
        return max_int(self.signed, self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def forward(
            self,
            scale: Tensor,
            zero_point: Tensor,
            bit_width: Tensor,
            x: Tensor) -> Tensor:
        y_int = self.to_int(scale, zero_point, bit_width, x)
        y = y_int - zero_point
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y


class DecoupledIntQuant(brevitas.jit.ScriptModule):
    """
    ScriptModule that implements scale, shifted, uniform integer quantization of an input tensor,
    according to an input pre-scale, scale, pre-zero-point, zero-point and bit-width.

    Args:
        narrow_range (bool): Flag that determines whether restrict quantization to a narrow range or not.
        signed (bool): Flag that determines whether to quantize to a signed range or not.
        float_to_int_impl (Module): Module that performs the conversion from floating point to
            integer representation. Default: RoundSte()
        tensor_clamp_impl (Module): Module that performs clamping. Default: TensorClamp()
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tensor: Quantized output in de-quantized format.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> int_quant = DecoupledIntQuant(narrow_range=True, signed=True)
        >>> scale, zero_point, bit_width = torch.tensor(0.01), torch.tensor(0.), torch.tensor(4.)
        >>> pre_scale, pre_zero_point = torch.tensor(0.02), torch.tensor(0.)
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out = int_quant(pre_scale, pre_zero_point, scale, zero_point, bit_width, inp)
        >>> out
        tensor([ 0.0200, -0.0300,  0.0700, -0.0700])

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    __constants__ = ['signed', 'narrow_range']

    def __init__(
            self,
            narrow_range: bool,
            signed: bool,
            float_to_int_impl: Module = RoundSte(),
            tensor_clamp_impl: Module = TensorClamp(),
            quant_delay_steps: int = 0):
        super(DecoupledIntQuant, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.tensor_clamp_impl = tensor_clamp_impl
        self.signed = signed
        self.narrow_range = narrow_range
        self.delay_wrapper = DelayWrapper(quant_delay_steps)

    @brevitas.jit.script_method_110_disabled
    def to_int(
            self,
            pre_scale: Tensor,
            pre_zero_point: Tensor,
            bit_width: Tensor,
            x: Tensor) -> Tensor:
        y = x / pre_scale
        y = y + pre_zero_point
        min_int_val = self.min_int(bit_width)
        max_int_val = self.max_int(bit_width)
        y = self.tensor_clamp_impl(y, min_val=min_int_val, max_val=max_int_val)
        y = self.float_to_int_impl(y)
        return y

    @brevitas.jit.script_method
    def min_int(self, bit_width):
        return min_int(self.signed, self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def max_int(self, bit_width):
        return max_int(self.signed, self.narrow_range, bit_width)

    @brevitas.jit.script_method
    def forward(
            self,
            pre_scale: Tensor,
            pre_zero_point: Tensor,
            scale: Tensor,
            zero_point: Tensor,
            bit_width: Tensor,
            x: Tensor) -> Tensor:
        y_int = self.to_int(pre_scale, pre_zero_point, bit_width, x)
        y = y_int - zero_point
        y = y * scale
        y = self.delay_wrapper(x, y)
        return y
