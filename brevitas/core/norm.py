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
from typing import Tuple, Optional, List

import torch
from torch.nn import Module, Parameter

import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.function.ops import min_int, max_int
from brevitas.utils.python_utils import AutoName
from .restrict_val import RestrictValue, RestrictValueType, FloatToIntImplType, RestrictValueOpImplType
from .stats import StatsOp, StatsInputViewShapeImpl, ParameterListStats

SCALING_SCALAR_SHAPE = ()
EPS = 1e-22


class NormImplType(AutoName):
    SAME_AS_SCALING = auto()
    MAX = auto()
    MAX_AVE = auto()
    MAX_L2 = auto()


class ParameterListNorm(torch.jit.ScriptModule):
    __constants__ = ['eps']

    def __init__(
            self,
            stats_op: StatsOp,
            input_view_shape_impl: StatsInputViewShapeImpl,
            output_shape: Tuple[int, ...],
            reduce_dim: Optional[int],
            input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter]):
        super(ParameterListNorm, self).__init__()
        assert stats_op == StatsOp.MAX or stats_op == StatsOp.MAX_AVE or StatsOp.MAX_L2

        if (stats_op == StatsOp.MAX_AVE or stats_op == StatsOp.MAX_L2) and output_shape != SCALING_SCALAR_SHAPE:
            raise Exception("Norm with MAX_AVE/MAX_L2 stats can't be over output channels.")
        self.eps = EPS
        self.parameter_list_stats = ParameterListStats(
            stats_op=stats_op,
            stats_output_shape=output_shape,
            stats_reduce_dim=reduce_dim,
            stats_input_view_shape_impl=input_view_shape_impl,
            stats_input_concat_dim=input_concat_dim,
            tracked_parameter_list=tracked_parameter_list,
            sigma=None)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, s: torch.Tensor):
        norm = self.parameter_list_stats()
        norm = norm + self.eps
        return norm


class SameAsScalingNorm(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, s: torch.Tensor):
        return s



