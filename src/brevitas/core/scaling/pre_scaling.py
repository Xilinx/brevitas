# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

import brevitas
import brevitas.config as config
from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.stats import SCALAR_SHAPE
from brevitas.core.stats.stats_wrapper import _Stats
from brevitas.core.zero_point import PreZeroCenterZeroPoint
from brevitas.function import abs_binary_sign_grad
from brevitas.function import get_upper_bound_on_l1_norm

__all__ = [
    "ParameterPreScalingWeightNorm",
    "AccumulatorAwareParameterPreScaling",
    "AccumulatorAwareZeroCenterParameterPreScaling"]


class ParameterPreScalingWeightNorm(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of learned pre-clipping scaling factor to support weight
    normalization-based quantization as proposed in `A2Q: Accumulator-Aware Quantization
    with Guaranteed Overflow Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig.

    The module parameterizes the pre-clipping scaling factor (i.e., `pre_scale`) of the
    decoupled tensor quantizer (i.e., `DecoupledRescalingIntQuant`) by combining the
    calculuated weight norm stats (i.e., `d_w`) with both the parameterized weight norm
    vector (i.e., `g`) and the post-clipping scaling factor (i.e., `post_scale`). The
    arithmetic is outlined below.

    The formulation for weight normalization-based quantization is given below:
        `y = clip(round( (g / s) * (w / norm(w)) )) * s`
    which we re-write as:
        `y = clip(round(w / pre_scale)) * post_scale`
    where `pre_scale = s * norm(w) / g` and `post_scale = s`.

    Here, `pre_scale` refers to the pre-clipping scaling factor and `post_scale` refers to
    the post-clipping scaling factor.

    Args:
        scaling_impl (Module): post-clipping scaling factor.
        normalize_stats_impl (Module): calculate statistics for normalizing weight parameter.
        scaling_stats_input_view_shape_impl (Module): transforming scaling to a new shape.
        tracked_parameter_list (List[torch.nn.Parameter]): list of tracked weight parameters
            for tensor quantizer.
        pre_scaling_shape (Tuple[int]): shape of pre-clipping scaling factor. Default: None.
        restrict_pre_scaling_impl (Module): restrict pre_scaling_init according to some
            criteria. Default: None.
        pre_scaling_min_val (float): force a lower-bound on scaling_init. Default: None.

    Returns:
        Tensor: scaling factor wrapped in a float torch.Tensor.
    """

    def __init__(
            self,
            scaling_impl: Module,
            normalize_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            tracked_parameter_list: List[torch.nn.Parameter],
            pre_scaling_shape: Optional[Tuple[int, ...]] = None,
            restrict_pre_scaling_impl: Optional[Module] = None,
            pre_scaling_min_val: Optional[float] = None) -> None:
        super(ParameterPreScalingWeightNorm, self).__init__()

        self.stats = _Stats(normalize_stats_impl, pre_scaling_shape)
        self.stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.scaling_impl = scaling_impl  # this is the post-clipping scaling factor

        if len(tracked_parameter_list) > 1:
            raise NotImplementedError(
                "Error: pre-clipping scales do not currently support multiple tracked quantizers.")
        assert len(tracked_parameter_list) == 1

        # Initialize the weight norm parameter vector from the tracked parameter itself
        param = tracked_parameter_list[0]
        param = self.stats_input_view_shape_impl(param)
        pre_scaling_init = self.stats(param)
        if restrict_pre_scaling_impl is not None:
            pre_scaling_init = restrict_pre_scaling_impl.restrict_init_tensor(pre_scaling_init)
        if pre_scaling_init.shape == SCALAR_SHAPE and pre_scaling_shape is not None:
            pre_scaling_init = torch.full(pre_scaling_shape, pre_scaling_init)
        self.value = Parameter(pre_scaling_init)
        self.restrict_clamp_scaling = _RestrictClampValue(
            pre_scaling_min_val, restrict_pre_scaling_impl)

    @brevitas.jit.script_method
    def forward(self, weights: Tensor) -> Tensor:
        """Takes weights as input and returns the pre-clipping scaling factor"""
        weights = self.stats_input_view_shape_impl(weights)
        d_w = self.stats(weights)  # denominator for weight normalization
        g = abs_binary_sign_grad(self.restrict_clamp_scaling(self.value))  # g
        s = self.scaling_impl(weights)  # s
        value = (s * d_w) / g
        return value

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(ParameterPreScalingWeightNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class AccumulatorAwareParameterPreScaling(ParameterPreScalingWeightNorm):
    """
    ScriptModule implementation of learned pre-clipping scaling factor to support
    accumulator-aware quantization (A2Q) as proposed in `A2Q: Accumulator-Aware Quantization
    with Guaranteed Overflow Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig.

    The module parameterizes the pre-clipping scaling factor (i.e., `pre_scale`) of the
    decoupled tensor quantizer (i.e., `DecoupledRescalingIntQuant`) by combining the
    calculated weight norm stats (i.e., `d_w`) with both the parameterized weight norm
    vector (i.e., `g`) and the post-clipping scaling factor (i.e., `post_scale`). The
    parameterized weight norm vector is then clamped to the upper bound on the l1-norm
    of the weights according to the accumulator bit-width bounds derived in the referenced
    paper. The pre-clipping scaling factor arithmetic is outlined below.

    The formulation for weight normalization-based quantization is given below:
        `y = clip(round( (g / s) * (w / norm(w)) )) * s`
    which we re-write as:
        `y = clip(round(w / pre_scale)) * post_scale`
    where `pre_scale = s * norm(w) / g` and `post_scale = s`.

    Here, `pre_scale` refers to the pre-clipping scaling factor and `post_scale` refers to
    the post-clipping scaling factor.

    Args:
        scaling_impl (Module): post-clipping scaling factor.
        normalize_stats_impl (Module): calculate statistics for normalizing weight parameter.
        accumulator_bit_width_impl (Module): module that returns the accumulator bit-width.
        scaling_stats_input_view_shape_impl (Module): transforming scaling to a new shape.
        tracked_parameter_list (List[torch.nn.Parameter]): list of tracked weight parameters
            for tensor quantizer.
        pre_scaling_shape (Tuple[int]): shape of pre-clipping scaling factor. Default: None.
        restrict_pre_scaling_impl (Module): restrict pre_scaling_init according to some
            criteria. Default: None.
        pre_scaling_min_val (float): force a lower-bound on scaling_init. Default: None.

    Returns:
        Tensor: scaling factor wrapped in a float torch.Tensor.
    """

    def __init__(
            self,
            scaling_impl: Module,
            normalize_stats_impl: Module,
            accumulator_bit_width_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            tracked_parameter_list: List[torch.nn.Parameter],
            pre_scaling_shape: Optional[Tuple[int, ...]] = None,
            restrict_pre_scaling_impl: Optional[Module] = None,
            pre_scaling_min_val: Optional[float] = None) -> None:
        super().__init__(
            scaling_impl,
            normalize_stats_impl,
            scaling_stats_input_view_shape_impl,
            tracked_parameter_list,
            pre_scaling_shape,
            restrict_pre_scaling_impl,
            pre_scaling_min_val)
        self.accumulator_bit_width = accumulator_bit_width_impl

    @brevitas.jit.script_method
    def calc_max_l1_norm(self, input_bit_width: Tensor, input_is_signed: bool) -> Tensor:
        accumulator_bit_width = self.accumulator_bit_width()
        upper_bound = get_upper_bound_on_l1_norm(
            accumulator_bit_width, input_bit_width, input_is_signed)
        return upper_bound

    @brevitas.jit.script_method
    def inner_forward(self, weights: Tensor, input_bit_width: Tensor, input_is_signed: bool):
        weights = self.stats_input_view_shape_impl(weights)
        d_w = self.stats(weights)  # denominator for weight normalization
        s = self.scaling_impl(weights)  # s
        g = abs_binary_sign_grad(self.restrict_clamp_scaling(self.value))  # g
        T = self.calc_max_l1_norm(input_bit_width, input_is_signed)  # T / s
        g = torch.clamp_max(g / s, T)
        value = d_w / g  # calculating final pre-clipping scaling factor
        # re-apply clamp_min_ste from restrict_scaling_impl to the specified pre_scaling_min_val
        value = self.restrict_clamp_scaling.clamp_min_ste(value)
        return value

    @brevitas.jit.script_method
    def forward(self, weights: Tensor, input_bit_width: Tensor, input_is_signed: bool) -> Tensor:
        """Takes weights, input bit-width, and input sign as input and returns the pre-clipping
        scaling factor per-channel, which is $s \cdot \Vert v \Vert_1 / g$"""
        value = self.inner_forward(weights, input_bit_width, input_is_signed)
        return value


class AccumulatorAwareZeroCenterParameterPreScaling(AccumulatorAwareParameterPreScaling):
    """
    ScriptModule implementation of learned pre-clipping scaling factor to support
    A2Q+ as proposed in `A2Q+: Improving Accumulator-Aware Weight Quantization`.

    The module implements the zero-centering constraint as a pre-clipping zero-point
    (i.e., `PreZeroCenterZeroPoint`) to relax the l1-norm constraint.

    Args:
        scaling_impl (Module): post-clipping scaling factor.
        pre_zero_point_impl (Module): pre-clipping zero-point.
        normalize_stats_impl (Module): calculate statistics for normalizing weight parameter.
        accumulator_bit_width_impl (Module): module that returns the accumulator bit-width.
        scaling_stats_input_view_shape_impl (Module): transforming scaling to a new shape.
        tracked_parameter_list (List[torch.nn.Parameter]): list of tracked weight parameters
            for tensor quantizer.
        pre_scaling_shape (Tuple[int]): shape of pre-clipping scaling factor. Default: None.
        restrict_pre_scaling_impl (Module): restrict pre_scaling_init according to some
            criteria. Default: None.
        pre_scaling_min_val (float): force a lower-bound on scaling_init. Default: None.

    Returns:
        Tensor: scaling factor wrapped in a float torch.Tensor.
    """

    def __init__(
            self,
            scaling_impl: Module,
            pre_zero_point_impl: Module,
            normalize_stats_impl: Module,
            accumulator_bit_width_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            tracked_parameter_list: List[Parameter],
            pre_scaling_shape: Optional[Tuple[int, ...]] = None,
            restrict_pre_scaling_impl: Optional[Module] = None,
            pre_scaling_min_val: Optional[float] = None) -> None:
        super().__init__(
            scaling_impl,
            normalize_stats_impl,
            accumulator_bit_width_impl,
            scaling_stats_input_view_shape_impl,
            tracked_parameter_list,
            pre_scaling_shape,
            restrict_pre_scaling_impl,
            pre_scaling_min_val)
        assert isinstance(
            pre_zero_point_impl, PreZeroCenterZeroPoint
        ), "Error: A2Q+ requires a pre-clipping zero-centering zero-point."
        self.pre_zero_point = pre_zero_point_impl

    @brevitas.jit.script_method
    def calc_max_l1_norm(self, input_bit_width: Tensor, input_is_signed: bool) -> Tensor:
        """ """
        assert input_bit_width is not None, "A2Q+ relies on input bit-width."
        max_accumulator_bit_width = self.accumulator_bit_width()  # P
        max_accumulator_mag = pow(2.0, max_accumulator_bit_width) - 2.0  # 2^P - 2
        max_input_mag = pow(2.0, input_bit_width) - 1.0  # 2^N - 1
        return max_accumulator_mag / max_input_mag

    @brevitas.jit.script_method
    def forward(self, weights: Tensor, input_bit_width: Tensor, input_is_signed: bool) -> Tensor:
        """Takes weights, input bit-width, and input sign as input and returns the pre-clipping
        scaling factor per-channel, which is $s \cdot \Vert v - \mu_v \Vert_1 / g$"""
        # NOTE: A2Q+ requires zero-centering the floating-point weights, which means that the
        # calculation of the l1-norm needs to be done over the zero-centered weights.
        z = self.pre_zero_point.get_zero_center(weights)
        value = self.inner_forward(weights + z, input_bit_width, input_is_signed)
        return value
