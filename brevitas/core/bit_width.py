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
from enum import auto

import torch
from torch import Tensor
from torch.nn import Parameter

import brevitas.config as config
from brevitas.utils.python_utils import AutoName
from brevitas.function.ops import tensor_clamp
from brevitas.function.ops_ste import tensor_clamp_ste
from .restrict_val import RestrictValueOpImplType, RestrictValueType, RestrictValue, FloatToIntImplType


MIN_INT_BIT_WIDTH = 2
NON_ZERO_EPSILON = 1e-6
REMOVE_ZERO_BIT_WIDTH = 0.1


class BitWidthImplType(AutoName):
    CONST = auto()
    PARAMETER = auto()


class ZeroLsbTruncBitWidth(torch.jit.ScriptModule):

    def forward(self, input_bit_width: Tensor, zero_hw_sentinel: Tensor):
        return zero_hw_sentinel


class BitWidthConst(torch.jit.ScriptModule):
    __constants__ = ['bit_width']

    def __init__(self, bit_width_init: int, restrict_bit_width_type: RestrictValueType) -> None:
        super(BitWidthConst, self).__init__()

        if restrict_bit_width_type != RestrictValueType.INT:
            raise Exception("When bit width is predefined, it has to be an INT value.")

        self.bit_width = int(bit_width_init)

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel: Tensor) -> Tensor:
        return self.bit_width + zero_hw_sentinel


class BitWidthParameter(torch.jit.ScriptModule):
    __constants__ = ['bit_width_base', 'max_bit_width', 'override_pretrained']

    def __init__(self,
                 bit_width_init: int,
                 min_overall_bit_width: Optional[int],
                 max_overall_bit_width: Optional[int],
                 restrict_bit_width_type: RestrictValueType,
                 override_pretrained: bool) -> None:
        super(BitWidthParameter, self).__init__()

        if min_overall_bit_width is None:
            min_overall_bit_width = MIN_INT_BIT_WIDTH
        if not (restrict_bit_width_type == RestrictValueType.FP
                or restrict_bit_width_type == RestrictValueType.INT
                or restrict_bit_width_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction on bit width {} not supported".format(restrict_bit_width_type))
        if bit_width_init < MIN_INT_BIT_WIDTH or min_overall_bit_width < MIN_INT_BIT_WIDTH:
            raise Exception("Int bit width has to be at least {}, instead is {}."
                            .format(MIN_INT_BIT_WIDTH, bit_width_init))

        self.override_pretrained = override_pretrained
        bit_width_init_op = RestrictValue.restrict_value_op(restrict_bit_width_type,
                                                            restrict_value_op_impl_type=RestrictValueOpImplType.MATH)
        self.restrict_bit_width = RestrictValue(restrict_bit_width_type,
                                                float_to_int_impl_type=FloatToIntImplType.ROUND,
                                                min_val=None)
        self.bit_width_base = bit_width_init_op(min_overall_bit_width)
        self.max_bit_width = bit_width_init_op(min_overall_bit_width) if max_overall_bit_width is not None else None
        bit_width_offset_init = max(bit_width_init_op(bit_width_init) - self.bit_width_base, 0.0)
        self.bit_width_offset = Parameter(torch.tensor(float(bit_width_offset_init)))

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel: Tensor) -> Tensor:
        if self.max_bit_width is not None:
            raise Exception("Not implemented yet.")
        bit_width = torch.abs(self.bit_width_offset) + self.bit_width_base
        bit_width = self.restrict_bit_width(bit_width)
        return bit_width

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        bit_width_offset_key = prefix + 'bit_width_offset'
        if self.override_pretrained and bit_width_offset_key in state_dict:
            del state_dict[bit_width_offset_key]
        super(BitWidthParameter, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bit_width_offset_key in missing_keys:
            missing_keys.remove(bit_width_offset_key)



class RemoveBitwidthParameter(torch.jit.ScriptModule):
    __constants__ = ['min_overall_bit_width', 'non_zero_epsilon', 'override_pretrained', 'remove_at_least_init_val']

    def __init__(self, bit_width_to_remove, remove_at_least_init_val, restrict_bit_width_impl, override_pretrained):
        super(RemoveBitwidthParameter, self).__init__()

        if bit_width_to_remove < 0:
            raise Exception("Bit width to clamp has to be at least 0, instead is {}."
                            .format(bit_width_to_remove))
        elif bit_width_to_remove == 0:
            bit_width_coeff_init = 1 / REMOVE_ZERO_BIT_WIDTH
        else:
            bit_width_coeff_init = 1 / bit_width_to_remove
        self.bit_width_coeff = Parameter(torch.tensor(bit_width_coeff_init))
        self.restrict_bit_width_impl = restrict_bit_width_impl
        self.non_zero_epsilon = NON_ZERO_EPSILON
        self.override_pretrained = override_pretrained
        self.remove_at_least_init_val = remove_at_least_init_val

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel) -> Tensor:
        bit_width_to_remove = 1.0 / (self.non_zero_epsilon + torch.abs(self.bit_width_coeff))
        bit_width_to_remove = self.restrict_bit_width_impl(bit_width_to_remove)
        return bit_width_to_remove

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        bit_width_coeff_key = prefix + 'bit_width_coeff'
        if self.override_pretrained and bit_width_coeff_key in state_dict:
            del state_dict[bit_width_coeff_key]
        super(RemoveBitwidthParameter, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                   missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bit_width_coeff_key in missing_keys:
            missing_keys.remove(bit_width_coeff_key)



class MsbClampParameterBitWidth(torch.jit.ScriptModule):
    __constants__ = ['min_overall_bit_width', 'max_overall_bit_width']

    def __init__(self,
                 ms_bit_width_to_clamp: int,
                 clamp_at_least_init_val: bool,
                 min_overall_bit_width: int,
                 max_overall_bit_width: int,
                 bit_width_impl_type: BitWidthImplType,
                 override_pretrained: bool) -> None:
        super(MsbClampParameterBitWidth, self).__init__()

        self.min_overall_bit_width = min_overall_bit_width
        self.max_overall_bit_width = max_overall_bit_width

        if bit_width_impl_type == BitWidthImplType.CONST:
            self.bit_width_to_remove_impl = BitWidthConst(ms_bit_width_to_clamp, RestrictValueType.INT)
        elif bit_width_impl_type == BitWidthImplType.PARAMETER:
            restrict_bit_width_impl = RestrictValue(RestrictValueType.INT,
                                                    float_to_int_impl_type=FloatToIntImplType.ROUND,
                                                    min_val=None)
            self.bit_width_to_remove_impl = RemoveBitwidthParameter(bit_width_to_remove=ms_bit_width_to_clamp,
                                                                    remove_at_least_init_val=clamp_at_least_init_val,
                                                                    restrict_bit_width_impl=restrict_bit_width_impl,
                                                                    override_pretrained=override_pretrained)
        else:
            raise Exception("Bit width implementation type {} not recognized for clamping accumulator."
                            .format(bit_width_impl_type))

    @torch.jit.script_method
    def forward(self, input_bit_width: Tensor, zero_hw_sentinel: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl(zero_hw_sentinel)
        output_bit_width = torch.abs(input_bit_width - bit_width_to_remove)
        output_bit_width = tensor_clamp_ste(output_bit_width,
                                            self.min_overall_bit_width + zero_hw_sentinel,
                                            self.max_overall_bit_width + zero_hw_sentinel) #todo STE on max only
        return output_bit_width


class LsbTruncParameterBitWidth(torch.jit.ScriptModule):
    __constants__ = ['is_const', 'min_overall_bit_width', 'max_overall_bit_width']

    def __init__(self,
                 ls_bit_width_to_trunc: int,
                 trunc_at_least_init_val: bool,
                 min_overall_bit_width: int,
                 max_overall_bit_width: int,
                 bit_width_impl_type: BitWidthImplType,
                 override_pretrained: bool):
        super(LsbTruncParameterBitWidth, self).__init__()

        self.min_overall_bit_width = min_overall_bit_width
        self.max_overall_bit_width = max_overall_bit_width

        if bit_width_impl_type == BitWidthImplType.CONST:
            self.bit_width_to_remove_impl = BitWidthConst(ls_bit_width_to_trunc, RestrictValueType.INT)
        elif bit_width_impl_type == BitWidthImplType.PARAMETER:
            restrict_bit_width_impl = RestrictValue(RestrictValueType.INT,
                                                    float_to_int_impl_type=FloatToIntImplType.ROUND,
                                                    min_val=None)
            self.bit_width_to_remove_impl = RemoveBitwidthParameter(bit_width_to_remove=ls_bit_width_to_trunc,
                                                                    remove_at_least_init_val=trunc_at_least_init_val,
                                                                    restrict_bit_width_impl=restrict_bit_width_impl,
                                                                    override_pretrained=override_pretrained)
        else:
            raise Exception("Bit width implementation type {} not recognized for truncating accumulator."
                            .format(bit_width_impl_type))

    @torch.jit.script_method
    def forward(self, input_bit_width: Tensor, zero_hw_sentinel: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl(zero_hw_sentinel)
        min_bit_width_to_remove = input_bit_width - self.max_overall_bit_width
        max_bit_width_to_remove = input_bit_width - self.min_overall_bit_width
        bit_width_to_remove = tensor_clamp(bit_width_to_remove,      # pass gradient to boundaries
                                           min_bit_width_to_remove,  # since input_bit_width is possibly learned
                                           max_bit_width_to_remove)
        return bit_width_to_remove