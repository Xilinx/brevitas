# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

import brevitas
from brevitas import config
from brevitas.core.stats import _ParameterListStats
from brevitas.core.stats import DEFAULT_MOMENTUM
from brevitas.core.stats import SCALAR_SHAPE
from brevitas.function import abs_binary_sign_grad

from .utils import inplace_momentum_update
from .utils import inplace_tensor_add
from .utils import StatelessBuffer

__all__ = [
    'ZeroZeroPoint',
    'StatsFromParameterZeroPoint',
    'ParameterFromRuntimeZeroPoint',
    'ParameterZeroPoint',
    'ParameterFromStatsFromParameterZeroPoint',
    'PreZeroCenterZeroPoint']


class ZeroZeroPoint(brevitas.jit.ScriptModule):

    def __init__(
            self,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ZeroZeroPoint, self).__init__()
        self.zero_point = StatelessBuffer(torch.tensor(0.0, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        return self.zero_point()


class _ScaleShiftZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['quantize_zero_point']

    def __init__(self, int_quant: Module, quantize_zero_point: bool) -> None:
        super(_ScaleShiftZeroPoint, self).__init__()
        self.int_quant = int_quant
        self.quantize_zero_point = quantize_zero_point

    @brevitas.jit.script_method
    def forward(self, zero_point: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        min_int = self.int_quant.min_int(bit_width)
        if self.quantize_zero_point:
            out = self.int_quant.to_int(scale, min_int, bit_width, zero_point)
        else:
            out = zero_point / scale + min_int
        return out


class _ScaleShiftQuantZeroPoint(brevitas.jit.ScriptModule):

    def __init__(self, zp_int_quant: Module) -> None:
        super(_ScaleShiftQuantZeroPoint, self).__init__()
        self.zp_int_quant = zp_int_quant

    @brevitas.jit.script_method
    def forward(self, zero_point: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        quant_zp, *_ = self.zp_int_quant(zero_point)
        return quant_zp


class StatsFromParameterZeroPoint(brevitas.jit.ScriptModule):

    def __init__(
            self,
            int_quant: Module,
            quantize_zero_point: bool,
            zero_point_stats_input_view_shape_impl: Module,
            zero_point_stats_input_concat_dim: int,
            zero_point_stats_impl: Module,
            zero_point_shape: Tuple[int, ...],
            tracked_parameter_list: List[torch.nn.Parameter],
            scale_shift_zero_point_impl: Optional[Module] = None) -> None:
        super(StatsFromParameterZeroPoint, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            zero_point_stats_impl,
            zero_point_shape,
            zero_point_stats_input_view_shape_impl,
            zero_point_stats_input_concat_dim,
            tracked_parameter_list)
        # This is for backward compatibility. Having int_quant/quantize_zero_point required for this
        # interface but not for the else seems a bit off and might require some clean-up.
        if scale_shift_zero_point_impl is None:
            self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)
        else:
            self.scale_shift_zero_point = scale_shift_zero_point_impl

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> torch.Tensor:
        stats = self.parameter_list_stats(x)
        return self.scale_shift_zero_point(-stats, scale, bit_width)


class ParameterFromRuntimeZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_permute_dims', 'zero_point_shape']

    def __init__(
            self,
            collect_stats_steps: int,
            int_quant: Module,
            quantize_zero_point: bool,
            zero_point_stats_impl: Optional[int],
            zero_point_shape: Tuple[int, ...],
            zero_point_stats_input_view_shape_impl: Module,
            zero_point_stats_momentum: Optional[float] = DEFAULT_MOMENTUM,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterFromRuntimeZeroPoint, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'
        self.collect_stats_steps = brevitas.jit.Attribute(collect_stats_steps, int)
        self.counter: int = brevitas.jit.Attribute(0, int)
        self.zero_point_shape = zero_point_shape
        self.stats_input_view_shape_impl = zero_point_stats_input_view_shape_impl
        self.momentum: Optional[float] = brevitas.jit.Attribute(
            zero_point_stats_momentum, Optional[float])
        self.value = Parameter(torch.full(zero_point_shape, 0.0, dtype=dtype, device=device))
        self.register_buffer(
            'buffer', torch.full(zero_point_shape, 0.0, dtype=dtype, device=device))
        self.zero_point_stats_impl = zero_point_stats_impl
        self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)
        self.local_loss_mode: bool = brevitas.jit.Attribute(False, bool)

    def init_zp(self):
        if self.counter <= self.collect_stats_steps:
            inplace_tensor_add(self.value.detach(), self.buffer)
            self.counter = self.collect_stats_steps + 1

    @brevitas.jit.script_method
    def training_forward(self, x) -> Tensor:
        if self.counter < self.collect_stats_steps:
            stats_input = self.stats_input_view_shape_impl(x)
            stats = self.zero_point_stats_impl(stats_input)
            stats = stats.view(self.zero_point_shape)
            if self.local_loss_mode:
                return stats
            new_counter = self.counter + 1
            if self.counter == 0:
                inplace_tensor_add(self.buffer, stats.detach())
            else:
                inplace_momentum_update(
                    self.buffer, stats.detach(), self.momentum, self.counter, new_counter)
            self.counter = new_counter
            # work around find_unusued_parameters=True in DDP
            out = stats + 0. * self.value
        elif self.counter == self.collect_stats_steps:
            self.init_zp()
            out = self.value
        else:
            out = self.value
        return out

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        if self.training:
            out = self.training_forward(x)
        else:
            if self.counter <= self.collect_stats_steps:
                out = self.buffer
            else:
                out = self.value
        out = abs_binary_sign_grad(out)
        out = self.scale_shift_zero_point(out, scale, bit_width)
        return out

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(ParameterFromRuntimeZeroPoint,
                            self).state_dict(destination, prefix, keep_vars)
        # Avoid saving the buffer
        del output_dict[prefix + 'buffer']
        # Avoid saving the init value
        if self.counter == 0:
            del output_dict[prefix + 'value']
        # Save buffer into value for any non-zero number of collection steps
        elif self.counter <= self.collect_stats_steps:
            output_dict[prefix + 'value'] = self.buffer
        return output_dict

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(ParameterFromRuntimeZeroPoint, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        buffer_key = prefix + 'buffer'
        # Buffer is supposed to be always missing
        missing_keys.remove(buffer_key)
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.counter = self.collect_stats_steps + 1
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ParameterZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_permute_dims', 'collect_stats_steps', 'momentum']

    def __init__(
            self,
            zero_point_init: Union[float, torch.Tensor],
            int_quant: Module,
            quantize_zero_point: bool,
            zero_point_shape: Tuple[int, ...] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterZeroPoint, self).__init__()
        if (isinstance(zero_point_init, Tensor) and zero_point_shape is not None and
                zero_point_init.shape != SCALAR_SHAPE and
                zero_point_init.shape != zero_point_shape):
            raise RuntimeError("zero_point_init.shape is non-scalar and != from zero_point_shape.")

        if isinstance(zero_point_init, Tensor):
            zero_point_init = zero_point_init.to(dtype=dtype, device=device)
            zero_point_init = zero_point_init.detach()
        else:
            zero_point_init = torch.tensor(zero_point_init, dtype=dtype, device=device)
        if zero_point_init.shape == SCALAR_SHAPE and zero_point_shape is not None:
            zero_point_init = torch.full(
                zero_point_shape, zero_point_init, dtype=dtype, device=device)
        self.value = Parameter(zero_point_init)
        self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        out = abs_binary_sign_grad(self.value)
        out = self.scale_shift_zero_point(out, scale, bit_width)
        return out

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(ParameterZeroPoint, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ParameterFromStatsFromParameterZeroPoint(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of a learned scale factor initialized from statistics of a parameter,
    e.g. weights MSE or AbsMax.
    """

    def __init__(
            self,
            int_quant: Module,
            quantize_zero_point: bool,
            zero_point_stats_input_view_shape_impl: Module,
            zero_point_stats_input_concat_dim: int,
            zero_point_stats_impl: Module,
            zero_point_shape: Tuple[int, ...],
            tracked_parameter_list: List[torch.nn.Parameter],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterFromStatsFromParameterZeroPoint, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            zero_point_stats_impl,
            zero_point_shape,
            zero_point_stats_input_view_shape_impl,
            zero_point_stats_input_concat_dim,
            tracked_parameter_list)
        self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)
        self.init_done: bool = brevitas.jit.Attribute(False, bool)
        self.local_loss_mode: bool = brevitas.jit.Attribute(False, bool)
        self.value = Parameter(torch.full(zero_point_shape, 0.0, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> torch.Tensor:
        if self.init_done:
            value = abs_binary_sign_grad(self.value)
            value = self.scale_shift_zero_point(value, scale, bit_width)
            return value
        else:
            stats = self.parameter_list_stats(x)
            # workaround to avoid find_ununsed_parameter=True in DDP
            stats = stats + 0. * self.value
            if self.local_loss_mode:
                return self.scale_shift_zero_point(-stats, scale, bit_width)
            inplace_tensor_add(self.value.detach(), stats)
            value = abs_binary_sign_grad(self.value)
            value = self.scale_shift_zero_point(value, scale, bit_width)
            self.init_done = True
            return value

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(ParameterFromStatsFromParameterZeroPoint, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Avoid saving the init value
        if not self.init_done and not config._FULL_STATE_DICT:
            del output_dict[prefix + 'value']
        return output_dict

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(ParameterFromStatsFromParameterZeroPoint, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.init_done = True
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class PreZeroCenterZeroPoint(brevitas.jit.ScriptModule):
    """Experimental ScriptModule implementation of a pre-scaling zero-point that zero-centers
    the incoming tensors. This is intended to be used with `DecoupledIntQuant`."""

    def __init__(
            self,
            stats_reduce_dim: int,
            pre_zero_point_stats_input_view_shape_impl: Module,
            pre_zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super(PreZeroCenterZeroPoint, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.stats_output_shape = pre_zero_point_shape
        self.stats_input_view_shape_impl = pre_zero_point_stats_input_view_shape_impl

    @brevitas.jit.script_method
    def get_zero_center(self, x: Tensor) -> Tensor:
        x = self.stats_input_view_shape_impl(x)
        u = torch.mean(x, dim=self.stats_reduce_dim, keepdim=True)
        z = -u.view(self.stats_output_shape)
        return z

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        # NOTE: `DecoupledIntQuant` adds the `pre_zero_point` value to the scaled tensor,
        # so this needs to return the negative of the scaled average value to perform
        # pre-zero centering before rounding and clipping
        z = self.get_zero_center(x) / scale  # need to scale the norm by s
        return z


class RuntimeDynamicGroupZeroPoint(brevitas.jit.ScriptModule):

    def __init__(
            self,
            input_view_impl: Module,
            zero_point_stats_impl: Module,
            int_quant: Module,
            quantize_zero_point: bool) -> None:
        super(RuntimeDynamicGroupZeroPoint, self).__init__()

        self.zero_point_stats_impl = zero_point_stats_impl
        self.input_view_impl = input_view_impl
        self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)

    @brevitas.jit.script_method
    def forward(self, stats_input: torch.Tensor, scale, bit_width) -> torch.Tensor:

        stats_input_reshaped = self.input_view_impl(stats_input)
        out = self.zero_point_stats_impl(stats_input_reshaped)
        return self.scale_shift_zero_point(-out, scale, bit_width)
