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

from typing import Tuple, Optional, List, Union
from enum import auto

import torch
from torch import nn
from torch.nn import Parameter

import brevitas.config as config
from brevitas.function.shape import *
from brevitas.core.function_wrapper import OverOutputChannelView, OverBatchOverTensorView
from brevitas.core.function_wrapper import OverBatchOverOutputChannelView, OverTensorView
from brevitas.utils.python_utils import AutoName

__all__ = ['StatsInputViewShapeImpl', 'StatsOp', 'ParameterListStats']


STD_DEV_EPSILON = 1e-8


class StatsInputViewShapeImpl(object):
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView


class StatsOp(AutoName):
    MAX = auto()
    AVE = auto()
    MAX_AVE = auto()
    MEAN_SIGMA_STD = auto()
    MEAN_LEARN_SIGMA_STD = auto()


class _ViewParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape']

    def __init__(self, parameter, view_shape_impl):
        super(_ViewParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl().shape(parameter)

    @torch.jit.script_method
    def forward(self):
        return self.parameter.view(self.shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_ViewParameterWrapper, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewParameterWrapper, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class _ViewCatParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape', 'cat_dim']

    def __init__(self, parameter, view_shape_impl, cat_dim):
        super(_ViewCatParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl().shape(parameter)
        self.cat_dim = cat_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.cat([self.parameter.view(self.shape), x], dim=self.cat_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_ViewCatParameterWrapper, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewCatParameterWrapper, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class AbsMax(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) -> None:
        super(AbsMax, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.reduce_dim is None:
            return torch.max(torch.abs(x))
        else:
            return torch.max(torch.abs(x), dim=self.reduce_dim)[0]


class AbsMaxAve(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) -> None:
        super(AbsMaxAve, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.mean(torch.max(torch.abs(x), dim=self.reduce_dim)[0])


class AbsAve(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim']

    def __init__(self, reduce_dim) -> None:
        super(AbsAve, self).__init__()
        self.reduce_dim = reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.reduce_dim is None:
            return torch.mean(torch.abs(x))
        else:
            return torch.mean(torch.abs(x), dim=self.reduce_dim)


class MeanSigmaStd(torch.jit.ScriptModule):
    __constants__ = ['reduce_dim', 'output_shape', 'std_dev_epsilon', 'const_sigma']

    def __init__(self, reduce_dim, const_sigma, learned_sigma, output_shape) -> None:
        super(MeanSigmaStd, self).__init__()
        self.reduce_dim = reduce_dim
        self.const_sigma = const_sigma
        self.learned_sigma = learned_sigma
        self.output_shape = output_shape
        self.std_dev_epsilon = STD_DEV_EPSILON

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        abs_val = torch.abs(x)
        if self.reduce_dim is None:
            mean_val = torch.mean(abs_val)
            std_val = torch.sqrt(torch.var(abs_val) + self.std_dev_epsilon)
        else:
            mean_val = torch.mean(torch.abs(x), dim=self.reduce_dim)
            mean_val = mean_val.view(self.output_shape)
            std_val = torch.sqrt(torch.var(abs_val, dim=self.reduce_dim) + self.std_dev_epsilon)
            std_val = std_val.view(self.output_shape)
        if self.const_sigma is not None:
            return mean_val + self.const_sigma * std_val
        else:
            return mean_val + self.learned_sigma * std_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(MeanSigmaStd, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        sigma_key = prefix + 'learned_sigma'
        if config.IGNORE_MISSING_KEYS and sigma_key in missing_keys:
            missing_keys.remove(sigma_key)


class Stats(torch.jit.ScriptModule):
    __constants__ = ['stats_output_shape',
                     'stats_reduce_dim']

    def __init__(self,
                 stats_op: StatsOp,
                 stats_reduce_dim: Optional[int],
                 stats_output_shape: Tuple[int, ...],
                 sigma: Optional[float]) -> None:
        super(Stats, self).__init__()

        if stats_reduce_dim is not None and len(stats_output_shape) < 2:
            raise Exception("Defining a reduce dimension requires the output view shape to have at least 2 dims.")
        if  len(stats_output_shape) > 1 and stats_reduce_dim is None:
            raise Exception("Defining an output view shape with more than 1 dims assumes a not None reduce dim.")
        if (stats_op == StatsOp.MEAN_SIGMA_STD or stats_op == StatsOp.MEAN_LEARN_SIGMA_STD) and sigma is None:
            raise Exception("Stats of type {} requires to define a value for sigma.".format(str(stats_op)))

        self.stats_output_shape = stats_output_shape

        if stats_op == StatsOp.MAX:
            self.stats_impl = AbsMax(reduce_dim=stats_reduce_dim)
        elif stats_op == StatsOp.AVE:
            self.stats_impl = AbsAve(reduce_dim=stats_reduce_dim)
        elif stats_op == StatsOp.MAX_AVE:
            self.stats_impl = AbsMaxAve(reduce_dim=stats_reduce_dim)
        elif stats_op == StatsOp.MEAN_SIGMA_STD or stats_op == StatsOp.MEAN_LEARN_SIGMA_STD:
            const_sigma = None
            learned_sigma = None
            if stats_op == StatsOp.MEAN_LEARN_SIGMA_STD:
                learned_sigma = Parameter(torch.full(stats_output_shape, sigma))
            else:
                const_sigma = sigma
            self.stats_impl = MeanSigmaStd(stats_reduce_dim, const_sigma, learned_sigma, stats_output_shape)
        else:
            raise Exception("Stats op {} not recognized".format(str(stats_op)))

    @torch.jit.script_method
    def forward(self, input) -> torch.Tensor:
        stats = self.stats_impl(input)
        stats = stats.view(self.stats_output_shape)
        return stats


class RuntimeStats(torch.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim',
                     'stats_permute_dims',
                     'momentum']

    def __init__(self,
                 stats_op: StatsOp,
                 stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 stats_permute_dims: Tuple[int, ...],
                 stats_reduce_dim: Optional[int],
                 stats_output_shape: Tuple[int, ...],
                 stats_buffer_momentum: float,
                 stats_buffer_init: float,
                 sigma: Optional[float]) -> None:
        super(RuntimeStats, self).__init__()

        self.stats_permute_dims = stats_permute_dims
        self.stats_input_view_shape_impl = stats_input_view_shape_impl()
        self.stats = Stats(stats_op=stats_op,
                           stats_output_shape=stats_output_shape,
                           stats_reduce_dim=stats_reduce_dim,
                           sigma=sigma)
        self.momentum = stats_buffer_momentum
        self.register_buffer('running_stats', torch.full(stats_output_shape, stats_buffer_init))

    @torch.jit.script_method
    def forward(self, stats_input) -> torch.Tensor:
        if self.stats_permute_dims is not None:
            stats_input = stats_input.permute(*self.stats_permute_dims).contiguous()
        stats_input = self.stats_input_view_shape_impl(stats_input)
        if self.training:
            out = self.stats(stats_input)
            self.running_stats *= (1 - self.momentum)
            self.running_stats += self.momentum * out.detach()
        else:
            out = self.running_stats
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(RuntimeStats, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        running_stats_key = prefix + 'running_stats'
        if config.IGNORE_MISSING_KEYS and running_stats_key in missing_keys:
            missing_keys.remove(running_stats_key)
        training_key = prefix + 'training' # Pytorch stores training flag as a buffer with JIT enabled
        if training_key in missing_keys:
            missing_keys.remove(training_key)


class ParameterListStats(torch.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim',
                     'extra_tracked_params_list']

    def __init__(self,
                 stats_op: StatsOp,
                 stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 stats_reduce_dim: Optional[int],
                 stats_input_concat_dim: int,
                 stats_output_shape: Tuple[int, ...],
                 tracked_parameter_list: List[torch.nn.Parameter],
                 sigma: Optional[float]) -> None:
        super(ParameterListStats, self).__init__()

        self.stats_input_concat_dim = stats_input_concat_dim
        self.first_tracked_param = _ViewParameterWrapper(tracked_parameter_list[0], stats_input_view_shape_impl)
        if len(tracked_parameter_list) > 1:
            extra_list = [_ViewCatParameterWrapper(param, stats_input_view_shape_impl, stats_input_concat_dim)
                          for param in tracked_parameter_list[1:]]
            self.extra_tracked_params_list = torch.nn.ModuleList(extra_list)
        else:
            self.extra_tracked_params_list = None
        self.stats = Stats(stats_op=stats_op,
                           stats_output_shape=stats_output_shape,
                           stats_reduce_dim=stats_reduce_dim,
                           sigma=sigma)

    @torch.jit.script_method
    def forward(self) -> torch.Tensor:
        stats_input = self.first_tracked_param()
        if self.extra_tracked_params_list is not None:
            for extra_tracked_param in self.extra_tracked_params_list:
                stats_input = extra_tracked_param(stats_input)
        out = self.stats(stats_input)
        return out



