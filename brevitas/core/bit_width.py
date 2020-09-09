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

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter, Module

import brevitas.config as config
from brevitas.function.ops import tensor_clamp
from brevitas.function.ops_ste import tensor_clamp_ste
from brevitas.inject.enum import BitWidthImplType  # retrocompatibility
from .utils import StatelessBuffer

assert BitWidthImplType  # prevent removal of unused import


MIN_INT_BIT_WIDTH = 2
NON_ZERO_EPSILON = 1e-6
REMOVE_ZERO_BIT_WIDTH = 0.1


class IdentityBitWidth(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        return x


class ZeroLsbTruncBitWidth(torch.jit.ScriptModule):

    def __init__(self) -> None:
        super(ZeroLsbTruncBitWidth, self).__init__()
        self.zero = StatelessBuffer(torch.tensor(0.0))

    def forward(self, input_bit_width: Tensor) -> Tensor:
        return self.zero()


class BitWidthConst(torch.jit.ScriptModule):

    def __init__(self, bit_width: int) -> None:
        super(BitWidthConst, self).__init__()
        assert isinstance(bit_width, int)
        self.bit_width = StatelessBuffer(torch.tensor(float(bit_width)))

    @torch.jit.script_method
    def forward(self) -> Tensor:
        return self.bit_width()


class BitWidthParameter(torch.jit.ScriptModule):
    __constants__ = ['bit_width_base', 'override_pretrained']

    def __init__(
            self,
            bit_width: int,
            restrict_bit_width_impl: Module,
            min_overall_bit_width: Optional[int] = None,
            override_pretrained_bit_width: bool = False) -> None:
        super(BitWidthParameter, self).__init__()

        if min_overall_bit_width is None:
            min_overall_bit_width = MIN_INT_BIT_WIDTH
        if bit_width < MIN_INT_BIT_WIDTH or min_overall_bit_width < MIN_INT_BIT_WIDTH:
            raise RuntimeError("Int bit width has to be at least {}, instead is {}."
                            .format(MIN_INT_BIT_WIDTH, bit_width))

        bit_width_base = restrict_bit_width_impl.restrict_init_float(min_overall_bit_width)
        bit_width = restrict_bit_width_impl.restrict_init_float(bit_width)
        bit_width_offset_init = max(bit_width - bit_width_base, 0.0)
        self.bit_width_offset = Parameter(torch.tensor(float(bit_width_offset_init)))
        self.bit_width_base = bit_width_base
        self.override_pretrained = override_pretrained_bit_width
        self.restrict_bit_width_impl = restrict_bit_width_impl

    @torch.jit.script_method
    def forward(self) -> Tensor:
        bit_width = torch.abs(self.bit_width_offset) + self.bit_width_base
        bit_width = self.restrict_bit_width_impl(bit_width)
        return bit_width

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        bit_width_offset_key = prefix + 'bit_width_offset'
        if self.override_pretrained and bit_width_offset_key in state_dict:
            del state_dict[bit_width_offset_key]
        super(BitWidthParameter, self)._load_from_state_dict(
            state_dict, prefix, local_metadata,strict,missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bit_width_offset_key in missing_keys:
            missing_keys.remove(bit_width_offset_key)


class RemoveBitwidthParameter(torch.jit.ScriptModule):
    __constants__ = ['non_zero_epsilon', 'override_pretrained']

    def __init__(
            self,
            bit_width_to_remove: int,
            override_pretrained_bit_width: bool = False,
            non_zero_epsilon: float = NON_ZERO_EPSILON,
            remove_zero_bit_width = REMOVE_ZERO_BIT_WIDTH):
        super(RemoveBitwidthParameter, self).__init__()

        if bit_width_to_remove < 0:
            raise RuntimeError("Bit width to clamp has to be >= 0.".format(bit_width_to_remove))
        elif bit_width_to_remove == 0:
            bit_width_coeff_init = 1 / remove_zero_bit_width
        else:
            bit_width_coeff_init = 1 / bit_width_to_remove
        self.bit_width_coeff = Parameter(torch.tensor(bit_width_coeff_init))
        self.non_zero_epsilon = non_zero_epsilon
        self.override_pretrained = override_pretrained_bit_width

    @torch.jit.script_method
    def forward(self) -> Tensor:
        bit_width_to_remove = 1.0 / (self.non_zero_epsilon + torch.abs(self.bit_width_coeff))
        return bit_width_to_remove

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        bit_width_coeff_key = prefix + 'bit_width_coeff'
        if self.override_pretrained and bit_width_coeff_key in state_dict:
            del state_dict[bit_width_coeff_key]
        super(RemoveBitwidthParameter, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bit_width_coeff_key in missing_keys:
            missing_keys.remove(bit_width_coeff_key)


class MsbClampBitWidth(torch.jit.ScriptModule):

    def __init__(
            self,
            bit_width_to_remove_impl: Module,
            min_overall_bit_width: int,
            max_overall_bit_width: int) -> None:
        super(MsbClampBitWidth, self).__init__()

        self.min_overall_bit_width = BitWidthConst(min_overall_bit_width)
        self.max_overall_bit_width = BitWidthConst(max_overall_bit_width)
        self.bit_width_to_remove_impl = bit_width_to_remove_impl

    @torch.jit.script_method
    def forward(self, input_bit_width: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl()
        output_bit_width = torch.abs(input_bit_width - bit_width_to_remove)
        # todo STE on max only
        output_bit_width = tensor_clamp_ste(
            output_bit_width,
            self.min_overall_bit_width(),
            self.max_overall_bit_width())
        return output_bit_width


class LsbTruncBitWidth(torch.jit.ScriptModule):

    def __init__(
            self,
            bit_width_to_remove_impl: Module,
            min_overall_bit_width: int,
            max_overall_bit_width: int):
        super(LsbTruncBitWidth, self).__init__()

        self.min_overall_bit_width = BitWidthConst(min_overall_bit_width)
        self.max_overall_bit_width = BitWidthConst(max_overall_bit_width)
        self.bit_width_to_remove_impl = bit_width_to_remove_impl

    @torch.jit.script_method
    def forward(self, input_bit_width: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl()
        min_bit_width_to_remove = input_bit_width - self.max_overall_bit_width()
        max_bit_width_to_remove = input_bit_width - self.min_overall_bit_width()
        # pass gradient to boundaries since input_bit_width is possibly learned
        bit_width_to_remove = tensor_clamp(
            bit_width_to_remove,
            min_bit_width_to_remove,
            max_bit_width_to_remove)
        return bit_width_to_remove
