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
from typing import Callable, Union, Optional
import math

import torch
from torch.nn import Sequential

from brevitas.utils.python_utils import AutoName
from .function_wrapper import RoundSte, CeilSte, Identity, PowerOfTwo, LogTwo, FloorSte, ClampMin


class RestrictValueType(AutoName):
    FP = auto()
    LOG_FP = auto()
    INT = auto()
    POWER_OF_TWO = auto()


class FloatToIntImplType(AutoName):
    ROUND = auto()
    CEIL = auto()
    FLOOR = auto()


class RestrictValueOpImplType(AutoName):
    MATH = auto()
    TORCH_FN = auto()
    TORCH_MODULE = auto()


class RestrictValue(torch.jit.ScriptModule):

    def __init__(self,
                 restrict_value_type: RestrictValueType,
                 float_to_int_impl_type: FloatToIntImplType,
                 min_val: Optional[float]) -> None:
        super(RestrictValue, self).__init__()

        if float_to_int_impl_type == FloatToIntImplType.ROUND:
            float_to_int_impl = RoundSte()
        elif float_to_int_impl_type == FloatToIntImplType.CEIL:
            float_to_int_impl = CeilSte()
        elif float_to_int_impl_type == FloatToIntImplType.FLOOR:
            float_to_int_impl = FloorSte()
        else:
            raise Exception("Float to int impl type {} not supported for restrict value"
                            .format(str(float_to_int_impl_type)))

        if min_val is not None:
            clamp_to_min_val = ClampMin(min_val=min_val)
        else:
            clamp_to_min_val = Identity()

        if restrict_value_type == RestrictValueType.FP:
            self.forward_impl = Sequential(Identity(), clamp_to_min_val)
        elif restrict_value_type == RestrictValueType.LOG_FP:
            self.forward_impl = Sequential(PowerOfTwo(), clamp_to_min_val)
        elif restrict_value_type == RestrictValueType.INT:
            self.forward_impl = Sequential(float_to_int_impl, clamp_to_min_val)
        elif restrict_value_type == RestrictValueType.POWER_OF_TWO:
            self.forward_impl = Sequential(float_to_int_impl, PowerOfTwo(), clamp_to_min_val)
        else:
            raise Exception("Restrict value type {} not recognized".format(str(restrict_value_type)))

    @staticmethod
    def restrict_value_op(restrict_value_type: RestrictValueType, restrict_value_op_impl_type: RestrictValueOpImplType):
        if restrict_value_type == RestrictValueType.FP or restrict_value_type == RestrictValueType.INT:
            return lambda x: x
        elif restrict_value_type == RestrictValueType.LOG_FP or restrict_value_type == RestrictValueType.POWER_OF_TWO:
            if restrict_value_op_impl_type == RestrictValueOpImplType.TORCH_FN:
                return torch.log2
            elif restrict_value_op_impl_type == RestrictValueOpImplType.MATH:
                return math.log2
            elif restrict_value_op_impl_type == RestrictValueOpImplType.TORCH_MODULE:
                return LogTwo()
            else:
                raise Exception("Type of implementation {} not recognized".format(str(restrict_value_op_impl_type)))
        else:
            raise Exception("Restriction of type {} not recognized".format(str(restrict_value_type)))

    @torch.jit.script_method
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.forward_impl(value)
        return value
