# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.restrict_val import SignedFloatRestrictValue
from brevitas.core.scaling.runtime import RuntimeDynamicGroupStatsScaling
from brevitas.core.scaling.runtime import RuntimeStatsScaling
from brevitas.core.scaling.runtime import StatsFromParameterScaling
from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.stats_op import SignedAbsMax
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.quant.solver.act import SolveActScalingImplFromEnum
from brevitas.quant.solver.common import SolveRestrictScaleSign
from brevitas.quant.solver.common import SolveRestrictScalingImplFromEnum
from brevitas.quant.solver.common import SolveScalingStatsOpFromEnum

SCALING_MIN_VAL = 1e-6


def test_scaling_min_val_parameter():
    inp = torch.zeros(1, 5, requires_grad=True)
    scaling_op = StatsFromParameterScaling(
        scaling_stats_impl=AbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_stats_input_concat_dim=None,
        tracked_parameter_list=[inp],
        scaling_shape=SCALAR_SHAPE,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_min_val=SCALING_MIN_VAL)
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()


def test_scaling_min_val_runtime():
    inp = torch.zeros(1, 5, requires_grad=True)
    scaling_op = RuntimeStatsScaling(
        scaling_stats_impl=AbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_shape=SCALAR_SHAPE,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_min_val=SCALING_MIN_VAL)
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()


def test_scaling_min_val_dynamic_group():
    inp = torch.zeros(1, 6, requires_grad=True)
    scaling_op = RuntimeDynamicGroupStatsScaling(
        group_size=3,
        group_dim=1,
        input_view_impl=Identity(),
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_stats_impl=AbsMax())
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()


def test_signed_scale_stats():
    scaling_op = RuntimeStatsScaling(
        scaling_stats_impl=SignedAbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        restrict_scaling_impl=SignedFloatRestrictValue(),
        scaling_shape=SCALAR_SHAPE,
        scaling_min_val=SCALING_MIN_VAL)
    inp = torch.tensor([-0.5, 0.0, 1.0])
    pre_scale = scaling_op(inp)
    assert pre_scale.item() == -1.


def test_signed_scale_stats_injector():

    class SignedStatsScaling(SolveActScalingImplFromEnum,
                             SolveScalingStatsOpFromEnum,
                             SolveRestrictScalingImplFromEnum,
                             SolveRestrictScaleSign):
        scaling_impl_type = ScalingImplType.STATS
        scaling_stats_op = StatsOp.SIGNED_MAX
        scaling_stats_input_view_shape_impl = Identity
        restrict_scaling_type = RestrictValueType.SIGNED_FP
        scaling_shape = SCALAR_SHAPE
        scaling_min_val = SCALING_MIN_VAL

    scaling_op = SignedStatsScaling.scaling_impl
    inp = torch.tensor([-0.5, 0.0, 1.0])
    pre_scale = scaling_op(inp)
    assert pre_scale.item() == -1.


def test_unsigned_scale_stats_injector_restrict_val_positive_scale():

    class SignedStatsScaling(SolveActScalingImplFromEnum,
                             SolveScalingStatsOpFromEnum,
                             SolveRestrictScalingImplFromEnum,
                             SolveRestrictScaleSign):
        scaling_impl_type = ScalingImplType.STATS
        scaling_stats_op = StatsOp.MAX
        scaling_stats_input_view_shape_impl = Identity
        restrict_scaling_type = RestrictValueType.FP
        scaling_shape = SCALAR_SHAPE
        scaling_min_val = SCALING_MIN_VAL

    scaling_op = SignedStatsScaling.scaling_impl
    inp = torch.tensor([-0.5, 0.0, 1.0])
    pre_scale = scaling_op(inp)
    assert isinstance(
        scaling_op.stats_scaling_impl.restrict_clamp_scaling.restrict_value_impl,
        FloatRestrictValue)
    assert pre_scale.item() == 1.


def test_signed_scale_stats_injector_restrict_val_positive_scale():

    class SignedStatsScaling(SolveActScalingImplFromEnum,
                             SolveScalingStatsOpFromEnum,
                             SolveRestrictScalingImplFromEnum,
                             SolveRestrictScaleSign):
        scaling_impl_type = ScalingImplType.STATS
        scaling_stats_op = StatsOp.SIGNED_MAX
        scaling_stats_input_view_shape_impl = Identity
        restrict_scaling_type = RestrictValueType.FP
        scaling_shape = SCALAR_SHAPE
        scaling_min_val = SCALING_MIN_VAL

    scaling_op = SignedStatsScaling.scaling_impl
    inp = torch.tensor([-0.5, 0.0, 1.0])
    pre_scale = scaling_op(inp)
    assert pre_scale.item() == 1.


def test_signed_scale_stats_restrict_val_po2_scale():
    scaling_op = RuntimeStatsScaling(
        scaling_stats_impl=SignedAbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_shape=SCALAR_SHAPE,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_min_val=SCALING_MIN_VAL)
    inp = torch.tensor([-0.5, 0.0, 1.0])
    pre_scale = scaling_op(inp)
    assert torch.all(torch.isnan(pre_scale))


def test_signed_scale_stats_injector_restrict_val_po2_scale():

    class SignedStatsScaling(SolveActScalingImplFromEnum,
                             SolveScalingStatsOpFromEnum,
                             SolveRestrictScalingImplFromEnum,
                             SolveRestrictScaleSign):
        scaling_impl_type = ScalingImplType.STATS
        scaling_stats_op = StatsOp.SIGNED_MAX
        scaling_stats_input_view_shape_impl = Identity
        restrict_scaling_type = RestrictValueType.POWER_OF_TWO
        scaling_shape = SCALAR_SHAPE
        scaling_min_val = SCALING_MIN_VAL

    # Verify that an exception is raised when using power of 2 scales
    # with a signed statistic
    with pytest.raises(ValueError, match=r"Statistic SignedAbsMax is signed*"):
        SignedStatsScaling.scaling_impl
