# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple, List, Optional

import torch
from torch import nn, Tensor

import brevitas
import brevitas.config as config
from .view_wrapper import _ViewCatParameterWrapper, _ViewParameterWrapper


DEFAULT_MOMENTUM = 0.1
SCALAR_SHAPE = ()


class _Stats(brevitas.jit.ScriptModule):
    __constants__ = ['stats_output_shape']

    def __init__(
            self,
            stats_impl: nn.Module,
            stats_output_shape: Tuple[int, ...]) -> None:
        super(_Stats, self).__init__()
        self.stats_output_shape = stats_output_shape
        self.stats_impl = stats_impl

    @brevitas.jit.script_method
    def forward(self, input: Tensor) -> Tensor:
        stats = self.stats_impl(input)
        stats = stats.view(self.stats_output_shape)
        return stats


class _RuntimeStats(brevitas.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim',
                     'stats_permute_dims',
                     'momentum']

    def __init__(
            self,
            stats_impl: nn.Module,
            stats_output_shape: Tuple[int, ...],
            stats_input_view_shape_impl: nn.Module,
            stats_buffer_momentum: float = DEFAULT_MOMENTUM) -> None:
        super(_RuntimeStats, self).__init__()
        self.first_batch = brevitas.jit.Attribute(True, bool)
        self.stats_input_view_shape_impl = stats_input_view_shape_impl
        self.stats = _Stats(stats_impl, stats_output_shape)
        self.momentum = stats_buffer_momentum
        self.register_buffer('running_stats', torch.full(stats_output_shape, 1.0))

    @brevitas.jit.script_method
    def forward(self, stats_input) -> Tensor:
        if self.training:
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


class _ParameterListStats(brevitas.jit.ScriptModule):
    __constants__ = ['stats_input_concat_dim']

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

    @brevitas.jit.script_method
    def forward(self) -> torch.Tensor:
        stats_input = self.first_tracked_param()
        if self.extra_tracked_params_list is not None:
            for extra_tracked_param in self.extra_tracked_params_list:
                stats_input = extra_tracked_param(stats_input)
        out = self.stats(stats_input)
        return out



