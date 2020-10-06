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

import math
from typing import Tuple, List, Optional

from torch import nn

import brevitas.config as config
from brevitas.function.shape import *
from brevitas.core.function_wrapper import OverOutputChannelView, OverBatchOverTensorView
from brevitas.core.function_wrapper import OverBatchOverOutputChannelView, OverTensorView
from brevitas.inject.enum import StatsOp  # noqa retrocompatibility

from .utils import StatelessBuffer

assert StatsOp  # prevent removal of unused import

DEFAULT_MOMENTUM = 0.1
DEFAULT_STD_DEV_EPSILON = 1e-8
SCALAR_SHAPE = ()


class StatsInputViewShapeImpl(object):
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView


class _ViewParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape']

    def __init__(self, parameter: nn.Parameter, view_shape_impl: nn.Module):
        super(_ViewParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl.shape(parameter)

    @torch.jit.script_method
    def forward(self):
        return self.parameter.view(self.shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_ViewParameterWrapper, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewParameterWrapper, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class _ViewCatParameterWrapper(torch.jit.ScriptModule):
    __constants__ = ['shape', 'cat_dim']

    def __init__(self, parameter: nn.Parameter, view_shape_impl: nn.Module, cat_dim: int):
        super(_ViewCatParameterWrapper, self).__init__()
        self.parameter = parameter
        self.shape = view_shape_impl.shape(parameter)
        self.cat_dim = cat_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.cat([self.parameter.view(self.shape), x], dim=self.cat_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_ViewCatParameterWrapper, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, dest=None, prefix='', keep_vars=False):
        output_dict = super(_ViewCatParameterWrapper, self).state_dict(dest, prefix, keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class AbsPercentile(torch.jit.ScriptModule):
    __constants__ = ['q', 'stats_reduce_dim']

    def __init__(self, percentile_q: float, stats_reduce_dim: Optional[int]):
        super(AbsPercentile, self).__init__()
        assert percentile_q <= 100, "q has to be a percentage"
        self.q = percentile_q
        self.stats_reduce_dim = stats_reduce_dim

    def forward(self, x):
        if self.stats_reduce_dim is None:
            # k is 1-indexed, so round away from zero
            k = math.floor(.01 * self.q * x.numel() + 0.5)
            result = x.abs().view(-1).kthvalue(k).values
        else:
            # assuming x is two dimensional, get the other dimension
            other_dim = abs(self.stats_reduce_dim - 1)
            dim_slice = torch.narrow(x, dim=other_dim, start=0, length=1)
            # k is 1-indexed, so round away from zero
            k = math.floor(.01 * self.q * dim_slice.numel() + 0.5)
            result = x.abs().kthvalue(k, dim=self.stats_reduce_dim).values
        return result


class AbsMax(torch.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int]) -> None:
        super(AbsMax, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.stats_reduce_dim is None:
            return torch.max(torch.abs(x))
        else:
            return torch.max(torch.abs(x), dim=self.stats_reduce_dim)[0]


class AbsMaxAve(torch.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int]) -> None:
        super(AbsMaxAve, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.mean(torch.max(torch.abs(x), dim=self.stats_reduce_dim)[0])


class AbsAve(torch.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int]) -> None:
        super(AbsAve, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.stats_reduce_dim is None:
            return torch.mean(torch.abs(x))
        else:
            return torch.mean(torch.abs(x), dim=self.stats_reduce_dim)


class MeanSigmaStd(torch.jit.ScriptModule):

    def __init__(
            self,
            stats_reduce_dim: int,
            sigma: float,
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(MeanSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, std_dev_epsilon)
        self.sigma = StatelessBuffer(torch.tensor(sigma))

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        sigma = self.sigma.view(self.sigma.shape)  # trick to get a tensor type
        out = self.impl(x, sigma)
        return out


class _MeanSigmaStdImpl(torch.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'output_shape', 'epsilon']

    def __init__(
            self,
            stats_reduce_dim: int,
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(_MeanSigmaStdImpl, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.epsilon = std_dev_epsilon

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        abs_val = torch.abs(x)
        if self.stats_reduce_dim is None:
            mean_val = torch.mean(abs_val)
            std_val = torch.sqrt(torch.var(abs_val) + self.std_dev_epsilon)
        else:
            mean_val = torch.mean(torch.abs(x), dim=self.stats_reduce_dim)
            std_val = torch.sqrt(torch.var(abs_val, dim=self.stats_reduce_dim) + self.epsilon)
            mean_val = mean_val.view(-1)
            std_val = std_val.view(-1)
        return mean_val + sigma * std_val


class MeanLearnedSigmaStd(torch.jit.ScriptModule):

    def __init__(
            self,
            stats_reduce_dim: int,
            sigma: float,
            stats_output_shape: Tuple[int, ...],
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(MeanLearnedSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, stats_output_shape, std_dev_epsilon)
        self.sigma = nn.Parameter(sigma)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        sigma = self.sigma.view(self.sigma.shape)  # trick to get a tensor type
        out = self.impl(x, sigma)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = prefix + 'sigma'
        retrocomp_value_key = prefix + 'learned_sigma'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(MeanLearnedSigmaStd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        sigma_key = prefix + 'sigma'
        if config.IGNORE_MISSING_KEYS and sigma_key in missing_keys:
            missing_keys.remove(sigma_key)


class _Stats(torch.jit.ScriptModule):
    __constants__ = ['stats_output_shape']

    def __init__(
            self,
            stats_impl: nn.Module,
            stats_output_shape: Tuple[int, ...]) -> None:
        super(_Stats, self).__init__()
        self.stats_output_shape = stats_output_shape
        self.stats_impl = stats_impl

    @torch.jit.script_method
    def forward(self, input) -> torch.Tensor:
        stats = self.stats_impl(input)
        stats = stats.view(self.stats_output_shape)
        return stats


class _RuntimeStats(torch.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim',
                     'stats_permute_dims',
                     'momentum']

    def __init__(
            self,
            stats_impl: nn.Module,
            stats_output_shape: Tuple[int, ...],
            stats_input_view_shape_impl: nn.Module,
            stats_permute_dims: Optional[Tuple[int, ...]] = None,
            stats_buffer_momentum: float = DEFAULT_MOMENTUM) -> None:
        super(_RuntimeStats, self).__init__()
        if stats_output_shape != SCALAR_SHAPE and stats_permute_dims is None:
            raise RuntimeError("Per channel runtime stats require a permute shape")
        self.first_batch = torch.jit.Attribute(True, bool)
        self.stats_permute_dims = stats_permute_dims
        self.stats_input_view_shape_impl = stats_input_view_shape_impl
        self.stats = _Stats(stats_impl, stats_output_shape)
        self.momentum = stats_buffer_momentum
        self.register_buffer('running_stats', torch.full(stats_output_shape, 1.0))

    @torch.jit.script_method
    def forward(self, stats_input) -> torch.Tensor:
        if self.training:
            if self.stats_permute_dims is not None:
                stats_input = stats_input.permute(*self.stats_permute_dims).contiguous()
            stats_input = self.stats_input_view_shape_impl(stats_input)
            out = self.stats(stats_input)
            if self.first_batch:
                self.running_stats *= out.detach()
                self.first_batch = False
            else:
                self.running_stats *= (1 - self.momentum)
                self.running_stats += self.momentum * out.detach()
        else:
            out = self.running_stats
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_RuntimeStats, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        running_stats_key = prefix + 'running_stats'
        if config.IGNORE_MISSING_KEYS and running_stats_key in missing_keys:
            missing_keys.remove(running_stats_key)
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)


class _ParameterListStats(torch.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim',
                     'extra_tracked_params_list']

    def __init__(
            self,
            stats_impl: nn.Module,
            stats_output_shape: Tuple[int, ...],
            stats_input_view_shape_impl: nn.Module,
            stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter]) -> None:
        super(_ParameterListStats, self).__init__()

        self.stats_input_concat_dim = stats_input_concat_dim
        self.first_tracked_param = _ViewParameterWrapper(
            tracked_parameter_list[0], stats_input_view_shape_impl)
        if len(tracked_parameter_list) > 1:
            extra_list = [
                _ViewCatParameterWrapper(param, stats_input_view_shape_impl, stats_input_concat_dim)
                          for param in tracked_parameter_list[1:]]
            self.extra_tracked_params_list = torch.nn.ModuleList(extra_list)
        else:
            self.extra_tracked_params_list = None
        self.stats = _Stats(stats_impl, stats_output_shape)

    @torch.jit.script_method
    def forward(self) -> torch.Tensor:
        stats_input = self.first_tracked_param()
        if self.extra_tracked_params_list is not None:
            for extra_tracked_param in self.extra_tracked_params_list:
                stats_input = extra_tracked_param(stats_input)
        out = self.stats(stats_input)
        return out



