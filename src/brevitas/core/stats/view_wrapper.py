# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

import brevitas
from brevitas.core.function_wrapper import StatsInputViewShapeImpl  # retrocomp


class _ViewParameterWrapper(brevitas.jit.ScriptModule):

    def __init__(self, parameter: Parameter, view_shape_impl: Module) -> None:
        super(_ViewParameterWrapper, self).__init__()
        self.parameter = parameter
        self.view_shape_impl = view_shape_impl

    @brevitas.jit.script_method
    def forward(self) -> Tensor:
        return self.view_shape_impl(self.parameter)

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(_ViewParameterWrapper, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewParameterWrapper, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict


class _ViewCatParameterWrapper(brevitas.jit.ScriptModule):
    __constants__ = ['cat_dim']

    def __init__(self, parameter: Parameter, view_shape_impl: Module, cat_dim: int) -> None:
        super(_ViewCatParameterWrapper, self).__init__()
        self.parameter = parameter
        self.view_shape_impl = view_shape_impl
        self.cat_dim = cat_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.view_shape_impl(self.parameter), x], dim=self.cat_dim)

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(_ViewCatParameterWrapper, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        parameter_key = prefix + 'parameter'
        if parameter_key in missing_keys:
            missing_keys.remove(parameter_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(_ViewCatParameterWrapper, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        del output_dict[prefix + 'parameter']
        return output_dict
