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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

import brevitas
from brevitas import config
from .utils import StatelessBuffer
from brevitas.core.stats import SCALAR_SHAPE, DEFAULT_MOMENTUM, NegativeMinOrZero
from brevitas.function import abs_binary_sign_grad
from brevitas.core.function_wrapper import RoundSte


class ZeroZeroPoint(brevitas.jit.ScriptModule):

    def __init__(self) -> None:
        super(ZeroZeroPoint, self).__init__()
        self.zero_point = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, min_int: Tensor) -> Tensor:
        return self.zero_point()


class MinUintZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, float_to_int_impl: Module, stats_reduce_dim: Optional[int]) -> None:
        super(MinUintZeroPoint, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.negative_min_or_zero = NegativeMinOrZero(stats_reduce_dim)

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, min_int: Tensor) -> Tensor:
        min_val = self.negative_min_or_zero(x)
        out = - self.float_to_int_impl(min_val / scale)
        out = min_int + out
        return out


class ParameterFromRuntimeMinZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_permute_dims',
                     'collect_stats_steps',
                     'momentum']

    def __init__(
            self,
            collect_stats_steps: int,
            float_to_int_impl: Module,
            stats_reduce_dim: Optional[int],
            zero_point_shape: Tuple[int, ...],
            zero_point_stats_input_view_shape_impl: Module,
            zero_point_stats_permute_dims: Optional[Tuple[int, ...]] = None,
            zero_point_stats_momentum: float = DEFAULT_MOMENTUM) -> None:
        super(ParameterFromRuntimeMinZeroPoint, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'
        if zero_point_shape != SCALAR_SHAPE and zero_point_stats_permute_dims is None:
            raise RuntimeError("Per channel runtime stats require a permute shape")
        self.collect_stats_steps = collect_stats_steps
        self.counter: int = brevitas.jit.Attribute(0, int)
        self.stats_permute_dims = zero_point_stats_permute_dims
        self.stats_input_view_shape_impl = zero_point_stats_input_view_shape_impl
        self.momentum = zero_point_stats_momentum
        self.value = Parameter(torch.full(zero_point_shape, 0.0))
        self.negative_min_or_zero = NegativeMinOrZero(stats_reduce_dim)
        self.float_to_int_impl = float_to_int_impl

    @brevitas.jit.script_method_110_disabled
    def forward(self, x: Tensor, scale: Tensor, min_int: Tensor) -> Tensor:
        if self.training:
            if self.counter <= self.collect_stats_steps:
                if self.stats_permute_dims is not None:
                    x = x.permute(*self.stats_permute_dims).contiguous()
                stats_input = self.stats_input_view_shape_impl(x)
                stats = self.negative_min_or_zero(stats_input)
                if self.counter == 0:
                    self.value.detach().add_(stats.detach())
                else:
                    self.value.detach().mul_(1 - self.momentum)
                    self.value.detach().add_(self.momentum * stats.detach())
                self.counter = self.counter + 1
                out = stats
            else:
                out = self.value
        else:
            out = self.value
        out = abs_binary_sign_grad(out)
        out = self.float_to_int_impl(out / scale)
        out = min_int + out
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ParameterFromRuntimeMinZeroPoint, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.counter = self.collect_stats_steps + 1


class ParameterZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_permute_dims',
                     'collect_stats_steps',
                     'momentum']

    def __init__(
            self,
            zero_point_init: Union[float, torch.Tensor],
            float_to_int_impl: Module = RoundSte(),
            zero_point_shape: Tuple[int, ...] = None) -> None:
        super(ParameterZeroPoint, self).__init__()
        if (isinstance(zero_point_init, Tensor)
                and zero_point_shape is not None
                and zero_point_init.shape != SCALAR_SHAPE
                and zero_point_init.shape != zero_point_shape):
            raise RuntimeError("zero_point_init.shape is non-scalar and != from zero_point_shape.")

        if isinstance(zero_point_init, Tensor):
            zero_point_init = zero_point_init.detach()
        else:
            zero_point_init = torch.tensor(zero_point_init)
        if zero_point_init.shape == SCALAR_SHAPE and zero_point_shape is not None:
            zero_point_init = torch.full(zero_point_shape, zero_point_init)
        self.value = Parameter(zero_point_init)
        self.float_to_int_impl = float_to_int_impl

    @brevitas.jit.script_method_110_disabled
    def forward(self, x: Tensor, scale: Tensor, min_int: Tensor) -> Tensor:
        out = abs_binary_sign_grad(self.value)
        out = self.float_to_int_impl(out / scale)
        out = min_int + out
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ParameterZeroPoint, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)

