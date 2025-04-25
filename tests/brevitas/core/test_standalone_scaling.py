import warnings

import pytest_cases
import torch

from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.scaling import FloatScaling
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling
from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE

SCALING_MIN_VAL = 1e-6


def test_scaling_state_dict():
    scaling_op = ParameterFromRuntimeStatsScaling(
        collect_stats_steps=10, scaling_stats_impl=AbsMax())

    with warnings.catch_warnings(record=True) as wlist:
        scaling_op.state_dict()
        for w in wlist:
            assert "Positional args are being deprecated" not in str(w.message)


@torch.no_grad()
def test_scaling_min_val_runtime():
    scaling_op = ParameterFromRuntimeStatsScaling(
        collect_stats_steps=1,
        scaling_stats_impl=AbsMax(),
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_impl=PowerOfTwoRestrictValue())
    inp = torch.zeros(1, 5)
    pre_scale = scaling_op(inp)
    value_scale_converted = scaling_op(inp)
    scaling_op.eval()
    assert not torch.isinf(scaling_op.value).any()


@torch.no_grad()
def test_scaling_min_val_param():
    inp = torch.zeros(1, 5)
    scaling_op = ParameterFromStatsFromParameterScaling(
        scaling_stats_impl=AbsMax(),
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_impl=PowerOfTwoRestrictValue(),
        scaling_stats_input_view_shape_impl=Identity(),
        scaling_stats_input_concat_dim=None,
        tracked_parameter_list=[inp],
        scaling_shape=SCALAR_SHAPE)
    pre_scale = scaling_op(inp)
    value_scale_converted = scaling_op(inp)
    assert not torch.isinf(scaling_op.value).any()


@torch.no_grad()
@pytest_cases.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_float_scaling_dtype(dtype):
    float_scaling_impl = FloatScaling(dtype=dtype)
    exp_bw = torch.tensor(4.)
    mant_bw = torch.tensor(3.)
    exp_bias = torch.tensor(0)
    float_scaling = float_scaling_impl(exp_bw, mant_bw, exp_bias)
    assert float_scaling.dtype == dtype
