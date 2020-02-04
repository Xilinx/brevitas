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
from brevitas.function.ops import min_int, max_int, max_uint, tensor_clamp, tensor_clamp_ste
from brevitas.function import binary_sign_ste, ternary_sign_ste


__all__ = ['QuantType', 'BinaryQuant', 'TernaryQuant', 'RescalingIntQuant',
           'PrescaledRestrictIntQuant']


class QuantType(AutoName):
    BINARY = auto()
    TERNARY = auto()
    INT = auto()
    FP = auto()


class IdentityQuant(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return x, zero_hw_sentinel, zero_hw_sentinel


class BinaryQuant(torch.jit.ScriptModule):
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(BinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = 1

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        y = binary_sign_ste(x) * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class ClampedBinaryQuant(torch.jit.ScriptModule):
    __constants__ = ['bit_width']

    def __init__(self, scaling_impl: Module):
        super(ClampedBinaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = 1
        
    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        y = tensor_clamp(x, - scale, scale)
        y = binary_sign_ste(y) * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class TernaryQuant(torch.jit.ScriptModule):
    __constants__ = ['threshold', 'bit_width']

    def __init__(self, scaling_impl: Module, threshold: float):
        super(TernaryQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.threshold = threshold
        self.bit_width = 2

    @torch.jit.script_method
    def forward(self, x: Tensor, zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(zero_hw_sentinel)
        mask = x.abs().ge(self.threshold * scale)
        y = mask.float() * ternary_sign_ste(x)
        y = y * scale
        return y, scale, zero_hw_sentinel + self.bit_width


class PrescaledRestrictIntQuantWithInputBitWidth(torch.jit.ScriptModule):

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

    @torch.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                input_bit_width: Tensor,
                zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(input_bit_width, zero_hw_sentinel)
        y = self.int_quant(scale, zero_hw_sentinel + 1, msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class PrescaledRestrictIntQuant(torch.jit.ScriptModule):

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

    @torch.jit.script_method
    def forward(self,
                x: Tensor,
                scale: Tensor,
                zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(zero_hw_sentinel)
        y = self.int_quant(scale, zero_hw_sentinel + 1, msb_clamp_bit_width, x)
        return y, scale, msb_clamp_bit_width


class IdentityPrescaledIntQuant(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x, input_scale, input_bit_width, zero_hw_sentinel) -> Tuple[Tensor, Tensor, Tensor]:
        return x, input_scale, input_bit_width


class RescalingIntQuant(torch.jit.ScriptModule):
    __constants__ = ['runtime']

    def __init__(self,
                 narrow_range: bool,
                 runtime: bool,
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
        self.runtime = runtime
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.msb_clamp_bit_width_impl = msb_clamp_bit_width_impl

    @staticmethod
    def scaling_init_from_min_max(min_val_init: Union[int, float], max_val_init: Union[int, float]) -> torch.Tensor:
        scaling_init = max(abs(float(min_val_init)), abs(float(max_val_init)))
        return torch.tensor(scaling_init)

    @torch.jit.script_method
    def forward(self,
                x: Tensor,
                zero_hw_sentinel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        msb_clamp_bit_width = self.msb_clamp_bit_width_impl(zero_hw_sentinel)
        if self.runtime:
            scale = self.scaling_impl(x)
        else:
            scale = self.scaling_impl(zero_hw_sentinel)
        int_scale = self.int_scaling_impl(msb_clamp_bit_width)
        y = self.int_quant(scale, int_scale, msb_clamp_bit_width, x)
        output_bit_width = msb_clamp_bit_width
        output_scale = scale / int_scale
        return y, output_scale, output_bit_width


class IntQuant(torch.jit.ScriptModule):
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
