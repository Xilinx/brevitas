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


from typing import Tuple, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module

import brevitas
import brevitas.config as config
from brevitas.core.utils import StatelessBuffer
from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.stats import _Stats, SCALAR_SHAPE, DEFAULT_MOMENTUM


class ConstScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_init: Union[float, Tensor],
            restrict_scaling_impl: Optional[Module] = None,
            scaling_min_val: Optional[float] = None) -> None:
        super(ConstScaling, self).__init__()
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        if isinstance(scaling_init, Tensor):
            if restrict_scaling_impl is not None:
                scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
            self.value = StatelessBuffer(scaling_init.detach())
        else:
            if restrict_scaling_impl is not None:
                scaling_init = restrict_scaling_impl.restrict_init_float(scaling_init)
            self.value = StatelessBuffer(torch.tensor(scaling_init))

    @brevitas.jit.script_method
    def forward(self, placeholder: Tensor) -> Tensor:
        value = self.value()
        restricted_value = self.restrict_clamp_scaling(value)
        return restricted_value


class ParameterScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_init: Union[float, Tensor],
            scaling_shape: Tuple[int, ...],
            restrict_scaling_impl: Module,
            scaling_min_val: Optional[float] = None) -> None:
        super(ParameterScaling, self).__init__()

        if isinstance(scaling_init, Tensor):
            scaling_init = scaling_init.detach()
        else:
            self.value = torch.tensor(scaling_init)
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
        if scaling_init.dim() == 0:
            self.value = Parameter(torch.full(scaling_shape, scaling_init))
        else:
            assert scaling_init.shape == scaling_shape
            self.value = Parameter(scaling_init)

    @brevitas.jit.script_method
    def forward(self, placeholder: Tensor) -> Tensor:
        value = self.restrict_clamp_scaling(self.value)
        return value

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(ParameterScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ParameterFromRuntimeStatsScaling(brevitas.jit.ScriptModule):
    __constants__ = ['stats_permute_dims',
                     'collect_stats_steps',
                     'momentum']

    def __init__(
            self,
            collect_stats_steps: int,
            restrict_scaling_impl: Module,
            scaling_stats_impl: Module,
            scaling_shape: Tuple[int, ...],
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_permute_dims: Optional[Tuple[int, ...]] = None,
            scaling_stats_momentum: float = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = None) -> None:
        super(ParameterFromRuntimeStatsScaling, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'
        if scaling_shape != SCALAR_SHAPE and scaling_stats_permute_dims is None:
            raise RuntimeError("Per channel runtime stats require a permute shape")
        self.collect_stats_steps = collect_stats_steps
        self.counter: int = brevitas.jit.Attribute(0, int)
        self.stats_permute_dims = scaling_stats_permute_dims
        self.stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats = _Stats(scaling_stats_impl, scaling_shape)
        self.momentum = scaling_stats_momentum
        self.value = Parameter(torch.full(scaling_shape, 1.0))
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        self.restrict_inplace_preprocess = restrict_scaling_impl.restrict_init_inplace_module()

    @brevitas.jit.script_method_110_disabled
    def forward(self, stats_input: Tensor) -> Tensor:
        if self.training:
            if self.counter < self.collect_stats_steps:
                if self.stats_permute_dims is not None:
                    stats_input = stats_input.permute(*self.stats_permute_dims).contiguous()
                stats_input = self.stats_input_view_shape_impl(stats_input)
                stats = self.stats(stats_input)
                if self.counter == 0:
                    self.value.detach().mul_(stats.detach())
                else:
                    self.value.detach().mul_(1 - self.momentum)
                    self.value.detach().add_(self.momentum * stats.detach())
                self.counter = self.counter + 1
                return stats
            elif self.counter == self.collect_stats_steps:
                self.restrict_inplace_preprocess(self.value.detach())
                self.counter = self.counter + 1
                return self.restrict_clamp_scaling(torch.abs(self.value))
            else:
                return self.restrict_clamp_scaling(torch.abs(self.value))
        out = self.restrict_clamp_scaling(torch.abs(self.value))
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ParameterFromRuntimeStatsScaling, self)._load_from_state_dict(
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