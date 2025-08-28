# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

import brevitas
import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper import OverBatchOverTensorView
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.restrict_val import _AbsValue
from brevitas.core.restrict_val import _ClampValue
from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.restrict_val import _RestrictValue
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.scaling.runtime import _StatsScaling
from brevitas.core.stats import _ParameterListStats
from brevitas.core.stats import _Stats
from brevitas.core.stats import DEFAULT_MOMENTUM
from brevitas.core.stats import SCALAR_SHAPE
from brevitas.core.utils import inplace_momentum_update
from brevitas.core.utils import inplace_tensor_mul
from brevitas.core.utils import StatelessBuffer


class ConstScaling(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of a constant scale factor.

    Args:
        scaling_init (Union[float, Tensor]): value to initialize the constant scale factor.
        is_scale_unsigned (bool): Whether the scale is unsigned. Default: True.
        restrict_scaling_impl (Module): restrict the scale factor according to some criteria. Default: FloatRestrictValue().
        restrict_threshold_impl (Optional[Module]): restrict the threshold according to some criteria. Default: None.
        restrict_scale_threshold_impl (Optional[Module]): restrict value of scale / threshold according to some criteria. Default: None.
        scaling_min_val (Optional[float]): force a lower-bound on the scale factor. Default: None.
        dtype (Optional[torch.dtype]): data type of the scale factor. Default: None.
        device (Optional[torch.device]): device of the scale factor. Default: None.

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
            is_scale_unsigned: bool = True,
            restrict_scaling_impl: Module = FloatRestrictValue(),
            restrict_threshold_impl: Optional[Module] = None,
            restrict_scale_threshold_impl: Optional[Module] = None,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ConstScaling, self).__init__()

        # Ensure retro-compatibility with shared threshold/scaling restrict
        if restrict_threshold_impl is None:
            restrict_threshold_impl = restrict_scaling_impl

        self.restrict_clamp_scaling = _RestrictClampValue(
            scaling_min_val, restrict_scaling_impl, is_scale_unsigned)
        self.restrict_clamp_threshold = _RestrictClampValue(
            restrict_value_impl=restrict_threshold_impl)
        self.restrict_clamp_scale_threshold = _RestrictClampValue(
            restrict_value_impl=restrict_scale_threshold_impl, is_unsigned=is_scale_unsigned)
        if isinstance(scaling_init, Tensor):
            scaling_init = scaling_init.to(device=device, dtype=dtype)
            scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
            self.value = StatelessBuffer(scaling_init.detach())
        else:
            scaling_init = restrict_scaling_impl.restrict_init_float(scaling_init)
            self.value = StatelessBuffer(torch.tensor(scaling_init, dtype=dtype, device=device))
        self.restrict_threshold_pre = restrict_threshold_impl.restrict_init_module()

    @brevitas.jit.script_method
    def forward(self, placeholder: Tensor, threshold: Optional[Tensor] = None) -> Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(placeholder)
        # We first apply any restriction to scaling
        # For IntQuant, this is no-op, retrocompatible.
        threshold = self.restrict_clamp_threshold(self.restrict_threshold_pre(threshold))
        restricted_value = self.restrict_clamp_scaling(self.value())
        restricted_value = self.restrict_clamp_scale_threshold(restricted_value / threshold)
        return restricted_value


class ParameterScaling(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of a learned scale factor.

    Args:
        scaling_init (Union[float, Tensor]): Value to initialize the learned scale factor.
        is_scale_unsigned (bool): Whether the scale is unsigned. Default: True.
        scaling_shape (Optional[Tuple[int, ...]]): Shape of the learned scale factor. Default: None.
        restrict_scaling_impl (Module): Restrict the scale factor according to some criteria. Default: FloatRestrictValue().
        restrict_threshold_impl (Optional[Module]): Restrict the threshold according to some criteria. Default: None.
        restrict_scale_threshold_impl (Optional[Module]): restrict value of scale / threshold according to some criteria. Default: None.
        scaling_min_val (Optional[float]): Force a lower-bound on the scale factor. Default: None.
        dtype (Optional[torch.dtype]): Data type of the scale factor. Default: None.
        device (Optional[torch.device]): Device of the scale factor. Default: None.

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
        Maps to scaling_impl_type == ScalingImplType.PARAMETER == 'PARAMETER' == 'parameter' in higher-level
    APIs.
    """

    def __init__(
            self,
            scaling_init: Union[float, Tensor],
            is_scale_unsigned: bool = True,
            scaling_shape: Optional[Tuple[int, ...]] = None,
            restrict_scaling_impl: Module = FloatRestrictValue(),
            restrict_threshold_impl: Optional[Module] = None,
            restrict_scale_threshold_impl: Optional[Module] = None,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterScaling, self).__init__()

        # Ensure retro-compatibility with shared threshold/scaling restrict
        if restrict_threshold_impl is None:
            restrict_threshold_impl = restrict_scaling_impl

        if (isinstance(scaling_init, Tensor) and scaling_shape is not None and
                scaling_init.shape != SCALAR_SHAPE and scaling_init.shape != scaling_shape):
            raise RuntimeError("scaling_init.shape is non-scalar and != from scaling_shape.")

        if isinstance(scaling_init, Tensor):
            scaling_init = scaling_init.to(device=device, dtype=dtype)
            scaling_init = scaling_init.detach()
        else:
            scaling_init = torch.tensor(scaling_init, dtype=dtype, device=device)

        scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)

        if scaling_init.shape == SCALAR_SHAPE and scaling_shape is not None:
            scaling_init = torch.full(scaling_shape, scaling_init, dtype=dtype, device=device)
        self.value = Parameter(scaling_init)
        self.restrict_clamp_scaling = _RestrictClampValue(
            scaling_min_val, restrict_scaling_impl, is_scale_unsigned)
        self.restrict_clamp_threshold = _RestrictClampValue(
            restrict_value_impl=restrict_threshold_impl)
        self.restrict_threshold_pre = restrict_threshold_impl.restrict_init_module()
        self.restrict_clamp_scale_threshold = _RestrictClampValue(
            restrict_value_impl=restrict_scale_threshold_impl, is_unsigned=is_scale_unsigned)

    @brevitas.jit.script_method
    def forward(self, placeholder: Tensor, threshold: Optional[Tensor] = None) -> Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(placeholder)
        # We first apply any restriction to scaling
        # For IntQuant, this is no-op, retrocompatible.
        threshold = self.restrict_clamp_threshold(self.restrict_threshold_pre(threshold))
        # We can clamp after restrict val since the learned parameter is already in log-domain
        value = self.restrict_clamp_scaling(self.value)
        value = self.restrict_clamp_scale_threshold(value / threshold)
        return value

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(ParameterScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ParameterFromStatsFromParameterScaling(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of a learned scale factor initialized from statistics computed over a list of parameters.

    Args:
        scaling_stats_impl (Module): Implementation of the statistics computed over the parameter list.
        scaling_stats_input_view_shape_impl (Module): Implementation of the view applied to the input before statistics computation.
        scaling_stats_input_concat_dim (int): Dimension along which to concatenate parameter tensors for statistics computation.
        tracked_parameter_list (List[torch.nn.Parameter]): List of parameters to track and compute statistics over.
        scaling_shape (Tuple[int, ...]): Shape of the learned scale factor.
        is_scale_unsigned (bool): Whether the scale is unsigned. Default: True.
        force_parameter (bool): If True, always use a tracked_parameter_list for statistics, even if only one is tracked. Default: False.
        restrict_scaling_impl (Module): Restrict the scale factor according to some criteria. Default: FloatRestrictValue().
        restrict_threshold_impl (Optional[Module]): Restrict the threshold according to some criteria. Default: None.
        restrict_scale_threshold_impl (Optional[Module]): restrict value of scale / threshold according to some criteria. Default: None.
        scaling_affine_rescaling_init (Optional[float]): Initial value for affine rescaling. Default: None.
        scaling_affine_shifting_init (Optional[float]): Initial value for affine shifting. Default: None.
        scaling_min_val (Optional[float]): Force a lower-bound on the scale factor. Default: None.
        dtype (Optional[torch.dtype]): Data type of the scale factor. Default: None.
        device (Optional[torch.device]): Device of the scale factor. Default: None.

    Returns:
        Tensor: learned scale factor wrapped in a float torch.tensor.

    Note:
        Set env variable BREVITAS_IGNORE_MISSING_KEYS=1 to avoid errors when retraining
        from a floating point state dict.

    Note:
        Maps to scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS ==
        'PARAMETER_FROM_STATS' == 'parameter_from_stats' in higher-level APIs.

    Example:
        >>> scaling_impl = ParameterFromStatsFromParameterScaling(
        ...     scaling_stats_impl=AbsMax(),
        ...     scaling_stats_input_view_shape_impl=Identity(),
        ...     scaling_stats_input_concat_dim=0,
        ...     tracked_parameter_list=[torch.nn.Parameter(torch.ones(3))],
        ...     scaling_shape=(3,))
        >>> x = torch.randn(3)
        >>> scaling_impl(x)
        tensor([...], grad_fn=<...>)
    """

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter],
            scaling_shape: Tuple[int, ...],
            is_scale_unsigned: bool = True,
            force_parameter: bool = False,
            restrict_scaling_impl: Module = FloatRestrictValue(),
            restrict_threshold_impl: Optional[Module] = None,
            restrict_scale_threshold_impl: Optional[Module] = None,
            scaling_affine_rescaling_init: Optional[float] = None,
            scaling_affine_shifting_init: Optional[float] = None,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterFromStatsFromParameterScaling, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_input_concat_dim,
            tracked_parameter_list,
            force_parameter)

        # Ensure retro-compatibility with shared threshold/scaling restrict
        if restrict_threshold_impl is None:
            restrict_threshold_impl = restrict_scaling_impl

        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl=restrict_scaling_impl,
            restrict_threshold_impl=restrict_threshold_impl,
            restrict_scale_threshold_impl=restrict_scale_threshold_impl,
            scaling_min_val=scaling_min_val,
            scaling_shape=scaling_shape,
            scaling_affine_rescaling_init=scaling_affine_rescaling_init,
            scaling_affine_shifting_init=scaling_affine_shifting_init,
            dtype=dtype,
            device=device,
            is_scale_unsigned=is_scale_unsigned)
        self.restrict_threshold_pre = restrict_threshold_impl.restrict_init_module()
        self.restrict_inplace_scaling_pre = restrict_scaling_impl.restrict_init_inplace_module()
        self.clamp_scaling = _ClampValue(scaling_min_val)

        self.init_done: bool = brevitas.jit.Attribute(False, bool)
        self.local_loss_mode: bool = brevitas.jit.Attribute(False, bool)

        self.value = Parameter(torch.full(scaling_shape, 1.0, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, threshold: Optional[Tensor] = None) -> Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(x)
        if self.init_done:
            threshold = self.stats_scaling_impl.restrict_clamp_threshold(
                self.restrict_threshold_pre(threshold))
            value = self.stats_scaling_impl.restrict_clamp_scaling(self.value)
            value = self.stats_scaling_impl.restrict_clamp_scale_threshold(value / threshold)
            return value
        else:
            stats = self.parameter_list_stats(x)
            # workaround to avoid find_ununsed_parameter=True in DDP
            stats = stats + 0. * self.value
            if self.local_loss_mode:
                # Scaling implementation before/after restrict_val is performed in stats_scaling_impl
                return self.stats_scaling_impl(stats, threshold)
            # Clamping avoids eventual log(0) with restrict_val
            stats = self.clamp_scaling(stats)
            stats = self.restrict_inplace_scaling_pre(stats)
            stats = self.stats_scaling_impl.affine_rescaling(stats)  # possible rescaling
            threshold = self.stats_scaling_impl.restrict_clamp_threshold(
                self.restrict_threshold_pre(threshold))
            inplace_tensor_mul(self.value.detach(), stats)
            value = self.stats_scaling_impl.restrict_clamp_scaling(self.value)
            value = self.stats_scaling_impl.restrict_clamp_scale_threshold(value / threshold)
            self.init_done = True
            return value

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(ParameterFromStatsFromParameterScaling, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Avoid saving the init value
        if not self.init_done and not config._FULL_STATE_DICT:
            del output_dict[prefix + 'value']
        return output_dict

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(ParameterFromStatsFromParameterScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.init_done = True
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
        is_scale_unsigned (bool, optional): Whether the scale is unsigned. Default: True.
        scaling_stats_input_view_shape_impl (Module, optional): Implementation of the view applied to the runtime
            input during the statistics collection phase. Default: OverBatchOverTensorView().
        scaling_shape (Tuple[int, ...], optional): Shape of the torch.nn.Parameter used in the second phase. Default: SCALAR_SHAPE.
        restrict_scaling_impl (Module, optional): Restrict the learned scale factor according to some criteria. Default: FloatRestrictValue().
        restrict_threshold_impl (Optional[Module], optional): Restrict the threshold according to some criteria. Default: None.
        restrict_scale_threshold_impl (Optional[Module]): restrict value of scale / threshold according to some criteria. Default: None.
        scaling_stats_momentum (Optional[float], optional): Momentum for the statistics moving average. Default: DEFAULT_MOMENTUM.
        scaling_min_val (Optional[float], optional): Force a lower-bound on the learned scale factor. Default: None.
        dtype (Optional[torch.dtype], optional): Data type of the scale factor. Default: None.
        device (Optional[torch.device], optional): Device of the scale factor. Default: None.

    Returns:
        Tensor: learned scale factor wrapped in a float torch.tensor.

    Raises:
        RuntimeError: if collect_stats_steps <= 0.

    Examples:
        >>> scaling_impl = ParameterFromRuntimeStatsScaling(
        ...     collect_stats_steps=1,
        ...     scaling_stats_impl=AbsMax())
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

    def __init__(
            self,
            collect_stats_steps: int,
            scaling_stats_impl: Module,
            is_scale_unsigned: bool = True,
            scaling_stats_input_view_shape_impl: Module = OverBatchOverTensorView(),
            scaling_shape: Tuple[int, ...] = SCALAR_SHAPE,
            restrict_scaling_impl: Module = FloatRestrictValue(),
            restrict_threshold_impl: Optional[Module] = None,
            restrict_scale_threshold_impl: Optional[Module] = None,
            scaling_stats_momentum: Optional[float] = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(ParameterFromRuntimeStatsScaling, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'

        # Ensure retro-compatibility with shared threshold/scaling restrict
        if restrict_threshold_impl is None:
            restrict_threshold_impl = restrict_scaling_impl

        self.collect_stats_steps: int = brevitas.jit.Attribute(collect_stats_steps, int)
        self.counter: int = brevitas.jit.Attribute(0, int)
        self.stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats = _Stats(scaling_stats_impl, scaling_shape)
        self.momentum: Optional[float] = brevitas.jit.Attribute(
            scaling_stats_momentum, Optional[float])
        self.register_buffer('buffer', torch.full(scaling_shape, 1.0, dtype=dtype, device=device))
        self.value = Parameter(torch.full(scaling_shape, 1.0, dtype=dtype, device=device))
        self.abs_value = _AbsValue(is_unsigned=is_scale_unsigned)
        self.restrict_scaling = _RestrictValue(restrict_scaling_impl)
        self.restrict_threshold = _RestrictValue(restrict_threshold_impl)
        self.restrict_scale_threshold = _RestrictValue(restrict_scale_threshold_impl)
        self.clamp_scaling = _ClampValue(scaling_min_val)
        self.local_loss_mode: bool = brevitas.jit.Attribute(
            False, bool)  # required to support MSE eval or variants
        self.restrict_inplace_preprocess = restrict_scaling_impl.restrict_init_inplace_module()
        self.restrict_scaling_pre = restrict_scaling_impl.restrict_init_module()
        self.restrict_threshold_pre = restrict_threshold_impl.restrict_init_module()

    def init_scale(self):
        if self.counter <= self.collect_stats_steps:
            self.restrict_inplace_preprocess(self.buffer)
            inplace_tensor_mul(self.value.detach(), self.buffer)
            self.counter = self.collect_stats_steps + 1

    @brevitas.jit.script_method
    def training_forward(self, stats_input: Tensor, threshold: Tensor) -> Tensor:
        if self.counter < self.collect_stats_steps:
            stats_input = self.stats_input_view_shape_impl(stats_input)
            stats = self.stats(stats_input)
            # workaround to avoid find_ununsed_parameter=True in DDP
            stats = stats + 0. * self.value  # stats gradient will change from None to 0.
            clamped_stats = self.clamp_scaling(stats)
            new_counter = self.counter + 1
            # Whenever we are in local loss mode, we don't update the counter nor the buffer
            if self.local_loss_mode:
                # Local loss mode, we early exit and divide by threshold
                return clamped_stats / threshold
            if self.counter == 0:
                inplace_tensor_mul(self.buffer, clamped_stats.detach())
            else:
                inplace_momentum_update(
                    self.buffer, clamped_stats.detach(), self.momentum, self.counter, new_counter)
            self.counter = new_counter
            return clamped_stats / threshold
        elif self.counter == self.collect_stats_steps:
            self.init_scale()
            value = self.clamp_scaling(self.restrict_scaling(self.value))
            threshold = self.restrict_threshold(self.restrict_threshold_pre(threshold))
            value = self.restrict_scale_threshold(value / threshold)
            return value
        else:
            threshold = self.restrict_threshold(self.restrict_threshold_pre(threshold))
            value = self.clamp_scaling(self.restrict_scaling(self.value))
            value = self.restrict_scale_threshold(value / threshold)
            return value

    @brevitas.jit.script_method
    def forward(self, stats_input: Tensor, threshold: Optional[Tensor] = None) -> Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(stats_input)
        if self.training:
            # Threshold division handled inside the training_forward
            return self.training_forward(stats_input, threshold)
        else:
            if self.counter <= self.collect_stats_steps:
                out = self.buffer
                # No clamping is necessary since statistics are already clamped in training_forward
                out = self.restrict_scaling_pre(out)
            else:
                out = self.value
            threshold = self.restrict_threshold(self.restrict_threshold_pre(threshold))
            out = self.restrict_scaling(out)
            out = self.abs_value(out)
            out = self.restrict_scale_threshold(out / threshold)
            # We can clamp after restrict val since the learned parameter is already in log-domain
            out = self.clamp_scaling(out)
        return out

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(ParameterFromRuntimeStatsScaling, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Avoid saving the buffer
        del output_dict[prefix + 'buffer']
        # Avoid saving the init value
        if self.counter == 0 and not config._FULL_STATE_DICT:
            del output_dict[prefix + 'value']
        # Save buffer into value for any non-zero number of collection steps
        elif self.counter <= self.collect_stats_steps:
            output_dict[prefix + 'value'] = self.restrict_scaling_pre(self.buffer)
        return output_dict

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        # Retrocompatibility with older ParameterScaling, for when scaling impl is switched over
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)

        super(ParameterFromRuntimeStatsScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # Buffer is supposed to be always missing
        missing_keys.remove(prefix + 'buffer')
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)
        # disable stats collection when a pretrained value is loaded
        if value_key not in missing_keys:
            self.counter = self.collect_stats_steps + 1
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class TruncMsbScaling(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of an integer scaling which calculates the scaling required to keep
    the most significant bits of the input. Interface compatible with
    :class:`~brevitas.core.quant.TruncIntQuant`'s `trunc_scaling_impl` member.

    Args:

    Returns:
        Tensor: truncation scale factor wrapped in a float torch.tensor.

    Examples:
        >>> from brevitas.core.scaling import TruncMsbScaling
        >>> trunc_scaling_impl = TruncMsbScaling()
        >>> input_bit_width, output_bit_width, signed = torch.tensor(8.), torch.tensor(4.), torch.tensor(True)
        >>> scaling_input = torch.Tensor([0.04, -0.05, 0.31, -0.44])
        >>> trunc_scale = trunc_scaling_impl(scaling_input, input_bit_width, output_bit_width, signed)
        >>> trunc_scale
        tensor(16.)

    Note:
        The forward method accepts a multiple placeholder arguments: `scaling_input` and `signed`
        to match the calling convention other `trunc_scaling_impl` modules. This is required by
        (early versions of) TorchScript to be consistent across different scaling implementations.

    Note:
        Maps to trunc_scaling_impl == TruncScalingImplType.MSB == 'MSB' == 'msb' in higher-level APIs.
    """

    def __init__(self) -> None:
        super(TruncMsbScaling, self).__init__()

    @brevitas.jit.script_method
    def forward(
            self,
            scaling_input: Tensor,
            input_bit_width: Tensor,
            output_bit_width: Tensor,
            signed: Union[bool, Tensor]) -> Tensor:
        return 2 ** (input_bit_width - output_bit_width)


class TruncScalingWrapper(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper which maps the inferface requirements of
    :class:`~brevitas.core.quant.TruncIntQuant`'s `trunc_scaling_impl` to standard scaling
    implementations through `scaling_impl`.

    Args:
        trunc_int_scaling_impl (Module): Module that takes in a bit-width and returns an integer scale
            factor, here interpreted as threshold on the integer range of quantization.
        scaling_impl (Module): Module that takes in the input to quantize and returns a scale factor,
            here interpreted as threshold on the floating-point range of quantization.
        tensor_clamp_impl (Module): Module that performs clamping. Default: TensorClamp()

    Returns:
        Tensor: truncation scale factor wrapped in a float torch.tensor.

    Examples:
        >>> from brevitas.core.scaling import TruncScalingWrapper
        >>> from brevitas.core.scaling import ConstScaling
        >>> from brevitas.core.scaling import PowerOfTwoIntScaling
        >>> trunc_scaling_impl = TruncScalingWrapper(PowerOfTwoIntScaling(), ConstScaling(1.))
        >>> input_bit_width, output_bit_width, signed = torch.tensor(8.), torch.tensor(4.), torch.tensor(True)
        >>> scaling_input = torch.Tensor([0.04, -0.05, 0.31, -0.44])
        >>> trunc_scale = trunc_scaling_impl(scaling_input, input_bit_width, output_bit_width, signed)
        >>> trunc_scale
        tensor(1.)

    Note:
        Maps to trunc_scaling_impl == TruncScalingImplType.WRAPPER == 'WRAPPER' == 'wrapper' in higher-level APIs.
    """

    def __init__(
        self,
        trunc_int_scaling_impl: Module,
        scaling_impl: Module,
        tensor_clamp_impl: Module = TensorClamp()) -> None:
        super(TruncScalingWrapper, self).__init__()
        self.trunc_int_scaling_impl = trunc_int_scaling_impl
        self.scaling_impl = scaling_impl
        self.tensor_clamp_impl = tensor_clamp_impl

    @brevitas.jit.script_method
    def forward(
            self,
            scaling_input: Tensor,
            input_bit_width: Tensor,
            output_bit_width: Tensor,
            signed: Union[bool, Tensor]) -> Tensor:
        threshold = self.trunc_int_scaling_impl(output_bit_width, signed)
        scale = self.scaling_impl(scaling_input, threshold)
        msb_scale = 2 ** (input_bit_width - output_bit_width)
        unit_scale = torch.ones_like(msb_scale)
        max_scale = torch.where(msb_scale > unit_scale, msb_scale, unit_scale)
        min_scale = torch.where(msb_scale < unit_scale, msb_scale, unit_scale)
        trunc_scale = self.tensor_clamp_impl(scale, min_scale, max_scale)
        return trunc_scale
