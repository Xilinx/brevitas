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

import warnings
from typing import Tuple, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module

import brevitas
import brevitas.config as config
from brevitas.function import abs_binary_sign_grad
from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper import OverBatchOverTensorView
from brevitas.core.utils import StatelessBuffer, inplace_momentum_update, inplace_tensor_mul
from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.stats import _Stats, SCALAR_SHAPE, DEFAULT_MOMENTUM


class ConstScaling(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of a constant scale factor.

    Args:
        scaling_init (Union[float, Tensor]): value to use as constant scale factor.
        restrict_scaling_impl (Module): restrict scaling_init according to some criteria. Default: None
        scaling_min_val (float): force a lower-bound on scaling_init. Default: None

    Returns:
        Tensor: scale factor wrapped in a float torch.tensor.

    Examples:
        >>> scaling_impl = ConstScaling(1.0)
        >>> scaling_impl(torch.empty(1))
        tensor(1.)
        >>> scaling_impl = ConstScaling(1.0, scaling_min_val=3.0)
        >>> scaling_impl(torch.empty(1))
        tensor(3.)
        >>> scaling_impl = ConstScaling(3.0, restrict_scaling_impl=PowerOfTwoRestrictValue())
        >>> scaling_impl(torch.empty(1))
        tensor(4.)

    Note:
        The forward method accepts a single placeholder argument. This is required by (early versions of)
        TorchScript to be consistent across different scaling implementations.

    Note:
        Maps to scaling_impl_type == ScalingImplType.CONST == 'CONST' == 'const' in higher-level APIs.
    """
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
    """
    ScriptModule implementation of a learned scale factor.

    Args:
        scaling_init (Union[float, Tensor]): value to initialize the learned scale factor.
        scaling_shape (Tuple[int, ...]): shape to extend a scalar float or tensor scaling_init. Default: None
        restrict_scaling_impl (Module): restrict the learned scale factor according to some criteria. Default: None
        scaling_min_val (float): force a lower-bound on the learned scale factor. Default: None

    Returns:
        Tensor: learned scale factor wrapped in a float torch.tensor.

    Raises:
        RuntimeError: if scaling_init is a non-scalar tensor and scaling_shape is != scaling_init.shape.

    Examples:
        >>> scaling_impl = ParameterScaling(6.0)
        >>> scaling_impl(torch.empty(1))
        tensor(6., grad_fn=<AbsBinarySignGradFnBackward>)
        >>> scaling_impl = ParameterScaling(6.0, scaling_shape=(3,))
        >>> scaling_impl(torch.empty(1))
        tensor([6., 6., 6.], grad_fn=<AbsBinarySignGradFnBackward>)
        >>> scaling_impl = ParameterScaling(6.0, scaling_shape=(3,), restrict_scaling_impl=PowerOfTwoRestrictValue())
        >>> scaling_impl(torch.empty(1))
        tensor([8., 8., 8.], grad_fn=<PowBackward1>)

    Note:
        Set env variable BREVITAS_IGNORE_MISSING_KEYS=1 to avoid errors when retraining
        from a floating point state dict.

    Note:
        The forward method accepts a single placeholder argument. This is required by (early versions of)
        TorchScript to be consistent across different scaling implementations.

    Note:
        Maps to scaling_impl_type == ScalingImplType.PARAMETER == 'PARAMETER' == 'parameter' in higher-level APIs.
    """
    def __init__(
            self,
            scaling_init: Union[float, Tensor],
            scaling_shape: Optional[Tuple[int, ...]] = None,
            restrict_scaling_impl: Optional[Module] = None,
            scaling_min_val: Optional[float] = None) -> None:
        super(ParameterScaling, self).__init__()

        if (isinstance(scaling_init, Tensor)
                and scaling_shape is not None
                and scaling_init.shape != SCALAR_SHAPE
                and scaling_init.shape != scaling_shape):
            raise RuntimeError("scaling_init.shape is non-scalar and != from scaling_shape.")

        if isinstance(scaling_init, Tensor):
            scaling_init = scaling_init.detach()
        else:
            scaling_init = torch.tensor(scaling_init)
        if restrict_scaling_impl is not None:
            scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
        if scaling_init.shape == SCALAR_SHAPE and scaling_shape is not None:
            scaling_init = torch.full(scaling_shape, scaling_init)
        self.value = Parameter(scaling_init)
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)

    @brevitas.jit.script_method
    def forward(self, placeholder: Tensor) -> Tensor:
        value = abs_binary_sign_grad(self.restrict_clamp_scaling(self.value))
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
    """
    ScriptModule implementation of a learned scale factor initialized from runtime statistics.
    The implementation works in two phases. During the first phase, statistics are collected in
    the same fashion as batchnorm, meaning that while the module is in training mode a set of per-batch
    statistics are computed and returned, while in background an average of them is retained and returned 
    in inference mode. During the second phase, the average accumulated during the first
    phase is used to initialize a learned torch.nn.Parameter, and then the behaviour is the same
    as ParameterScaling.

    Args:
        collect_stats_steps (int): Number of calls to the forward method in training mode to collect statistics for.
        scaling_stats_impl (Module): Implementation of the statistics computed during the collection phase.
        scaling_stats_input_view_shape_impl (Module): Implementation of the view applied to the runtime
            input during the statistics collection phase. Default: OverBatchOverTensorView().
        scaling_shape (Tuple[int, ...]): shape of the torch.nn.Parameter used in the second phase. Default: SCALAR_SHAPE.
        restrict_scaling_impl (Module): restrict the learned scale factor according to some criteria. Default: None
            input before going into scaling_stats_input_view_shape_impl. Default: None
        scaling_stats_momentum: float = Momentum for the statistics moving average. Default: DEFAULT_MOMENTUM.
        scaling_min_val (float): force a lower-bound on the learned scale factor. Default: None.

    Returns:
        Tensor: learned scale factor wrapped in a float torch.tensor.

    Raises:
        RuntimeError: if scaling_shape != SCALAR_SHAPE and scaling_stats_permute_dims is None

    Examples:
        >>> scaling_impl = ParameterFromRuntimeStatsScaling(collect_stats_steps=1, scaling_stats_impl=AbsMax())
        >>> scaling_impl.training
        True
        >>> x = torch.arange(-3, 2, 0.1)
        >>> scaling_impl(x)
        tensor(3.)
        >>> scaling_impl(torch.randn_like(x))
        tensor(3., grad_fn=<AbsBinarySignGradFnBackward>)

    Note:
        Set env variable BREVITAS_IGNORE_MISSING_KEYS=1 to avoid errors when retraining
        from a floating point state dict.

    Note:
        Maps to scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS == 'PARAMETER_FROM_STATS'
        == 'parameter_from_stats' when applied to runtime values (inputs/outputs/activations) in higher-level APIs.
    """
    __constants__ = ['collect_stats_steps', 'momentum']

    def __init__(
            self,
            collect_stats_steps: int,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module = OverBatchOverTensorView(),
            scaling_shape: Tuple[int, ...] = SCALAR_SHAPE,
            restrict_scaling_impl: Optional[Module] = None,
            scaling_stats_momentum: Optional[float] = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = None) -> None:
        super(ParameterFromRuntimeStatsScaling, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'
        self.collect_stats_steps = collect_stats_steps
        self.counter: int = brevitas.jit.Attribute(0, int)
        self.stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats = _Stats(scaling_stats_impl, scaling_shape)
        self.momentum = scaling_stats_momentum
        self.register_buffer('buffer', torch.full(scaling_shape, 1.0))
        self.value = Parameter(torch.full(scaling_shape, 1.0))
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        if restrict_scaling_impl is not None:
            self.restrict_inplace_preprocess = restrict_scaling_impl.restrict_init_inplace_module()
            self.restrict_preprocess = restrict_scaling_impl.restrict_init_module()
        else:
            self.restrict_inplace_preprocess = Identity()
            self.restrict_preprocess = Identity()
    
    @brevitas.jit.script_method
    def training_forward(self, stats_input: Tensor) -> Tensor:
        if self.counter < self.collect_stats_steps:
            stats_input = self.stats_input_view_shape_impl(stats_input)
            stats = self.stats(stats_input)
            new_counter = self.counter + 1
            if self.counter == 0:
                inplace_tensor_mul(self.buffer, stats.detach())
            else:
                inplace_momentum_update(
                    self.buffer, stats.detach(), self.momentum, self.counter, new_counter)
            self.counter = new_counter
            # workaround to avoid find_ununsed_parameter=True in DDP
            stats = stats + 0. * self.value
            return stats
        elif self.counter == self.collect_stats_steps:
            self.restrict_inplace_preprocess(self.buffer)
            inplace_tensor_mul(self.value.detach(), self.buffer)
            self.counter = self.counter + 1
            return abs_binary_sign_grad(self.restrict_clamp_scaling(self.value))
        else:
            return abs_binary_sign_grad(self.restrict_clamp_scaling(self.value))

    @brevitas.jit.script_method
    def forward(self, stats_input: Tensor) -> Tensor:
        if self.training:
            return self.training_forward(stats_input)
        else:
            if self.counter <= self.collect_stats_steps:
                out = self.buffer
                out = self.restrict_preprocess(out)
            else:
                out = self.value
            out = abs_binary_sign_grad(self.restrict_clamp_scaling(out))
        return out

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(ParameterFromRuntimeStatsScaling, self).state_dict(
            destination, prefix, keep_vars)        
        # Avoid saving the buffer
        del output_dict[prefix + 'buffer']
        # Avoid saving the init value
        if self.counter == 0:
            del output_dict[prefix + 'value']
        # Save buffer into value for any non-zero number of collection steps
        elif self.counter <= self.collect_stats_steps:
            output_dict[prefix + 'value'] = self.restrict_preprocess(self.buffer)
        return output_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ParameterFromRuntimeStatsScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        # Buffer is supposed to be always missing
        missing_keys.remove(prefix + 'buffer')
        # Retrocompatibility with older ParameterScaling, for when scaling impl is switched over
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.counter = self.collect_stats_steps + 1
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)
