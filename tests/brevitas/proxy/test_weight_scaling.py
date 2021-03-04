import pytest
from torch import nn

from brevitas.nn import QuantLinear
from brevitas import config


def test_parameter_from_stats_update():
    config.IGNORE_MISSING_KEYS = True
    linear = nn.Linear(10, 5, bias=False)
    q_linear = QuantLinear(
        10, 5, bias=False,
        weight_quant_type='binary',
        weight_scaling_impl_type='parameter_from_stats')
    l_max = linear.weight.abs().max()
    old_scale = q_linear.quant_weight_scale()
    old_ql_max = q_linear.weight.abs().max()
    q_linear.load_state_dict(linear.state_dict())
    new_scale = q_linear.quant_weight_scale()
    new_ql_max = q_linear.weight.abs().max()
    assert old_scale == old_ql_max
    assert new_scale == l_max
    assert new_scale == new_ql_max


def test_parameter_from_stats_state_dict():
    q_linear1 = QuantLinear(
        10, 5, bias=False,
        weight_quant_type='binary',
        weight_scaling_impl_type='parameter',
        weight_scaling_init=0.1)
    q_linear2 = QuantLinear(
        10, 5, bias=False,
        weight_quant_type='binary',
        weight_scaling_impl_type='parameter',
        weight_scaling_init=0.001)
    q_linear1_old_scale = q_linear1.quant_weight_scale()
    q_linear1.load_state_dict(q_linear2.state_dict())
    q_linear1_new_scale = q_linear1.quant_weight_scale()
    q_linear2_scale = q_linear2.quant_weight_scale()
    assert q_linear1_old_scale != q_linear2_scale
    assert q_linear1_old_scale != q_linear1_new_scale
    assert q_linear1_new_scale == q_linear2_scale
