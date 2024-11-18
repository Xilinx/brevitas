import torch

from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.scaling.runtime import RuntimeDynamicGroupStatsScaling
from brevitas.core.scaling.runtime import RuntimeStatsScaling
from brevitas.core.scaling.runtime import StatsFromParameterScaling
from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE


def test_scaling_min_val_parameter():
    inp = torch.zeros(1, 5, requires_grad=True)
    scaling_min_val = torch.tensor(1e-6)
    scaling_op = StatsFromParameterScaling(
        scaling_stats_impl=AbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_stats_input_concat_dim=None,
        tracked_parameter_list=[inp],
        scaling_shape=SCALAR_SHAPE,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_min_val=scaling_min_val)
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()


def test_scaling_min_val_runtime():
    inp = torch.zeros(1, 5, requires_grad=True)
    scaling_min_val = torch.tensor(1e-6)
    scaling_op = RuntimeStatsScaling(
        scaling_stats_impl=AbsMax(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_shape=SCALAR_SHAPE,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_min_val=scaling_min_val)
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()


def test_scaling_min_val_dynamic_group():
    inp = torch.zeros(1, 6, requires_grad=True)
    scaling_min_val = torch.tensor(1e-6)
    scaling_op = RuntimeDynamicGroupStatsScaling(
        group_size=3,
        group_dim=1,
        input_view_impl=Identity(),
        scaling_min_val=scaling_min_val,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_stats_impl=AbsMax())
    pre_scale = scaling_op(inp)
    pre_scale.sum().backward()
    assert not torch.isnan(inp.grad).any()
