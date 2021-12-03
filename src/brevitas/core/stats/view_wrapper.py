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


import torch
from torch import Tensor
from torch.nn import Parameter, Module

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