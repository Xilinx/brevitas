# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PiSO grid builders, proxy-accessor grid helpers, stored-scale
accessors and ordering helper (brevitas.graph.piso.utils)."""
import pytest
import torch

from brevitas.graph.piso.utils import _build_grid
from brevitas.graph.piso.utils import _get_base_scale
from brevitas.graph.piso.utils import _grid_threshold
from brevitas.graph.piso.utils import _has_base_scale
from brevitas.graph.piso.utils import _has_zero_zero_point
from brevitas.graph.piso.utils import _is_float_weight_quant
from brevitas.graph.piso.utils import _is_int_weight_quant
from brevitas.graph.piso.utils import _set_base_scale
from brevitas.graph.piso.utils import compute_group_importance_permutation
from brevitas.graph.piso.utils import make_fp_grid
from brevitas.graph.piso.utils import make_int_grid
from brevitas.inject.enum import ScalingImplType
import brevitas.nn as qnn
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE


@pytest.mark.parametrize("bit_width", [2, 3, 4, 8])
def test_make_int_grid_signed(bit_width):
    grid = make_int_grid(bit_width=bit_width, signed=True)
    lo = -(1 << (bit_width - 1))
    hi = (1 << (bit_width - 1)) - 1
    assert grid[0] == float(lo)
    assert grid[-1] == float(hi)
    assert len(grid) == (1 << bit_width)
    # sorted ascending, unique, contains 0
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)
    assert 0.0 in grid


@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_make_int_grid_narrow_range(bit_width):
    grid = make_int_grid(bit_width=bit_width, signed=True, narrow_range=True)
    # narrow range drops the extra negative value -> symmetric grid
    assert grid[0] == -grid[-1]
    assert 0.0 in grid


@pytest.mark.parametrize("e_bits,m_bits", [(2, 1), (3, 2), (2, 2)])
def test_make_fp_grid_properties(e_bits, m_bits):
    grid = make_fp_grid(e_bits=e_bits, m_bits=m_bits, signed=True)
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)
    assert 0.0 in grid
    # signed grid is symmetric about 0
    assert grid[0] == -grid[-1]


def test_compute_group_importance_permutation_basic():
    # diag(H) with clear per-block importance; blocks of size 2.
    # block 0 (idx 0,1) score 1+1=2 ; block 1 (idx 2,3) score 10+8=18
    x = torch.tensor([1.0, 1.0, 10.0, 8.0])
    perm, block_order = compute_group_importance_permutation(x, group_size=2)
    # descending block score -> block 1 first, then block 0
    assert block_order.tolist() == [1, 0]
    # perm keeps each block contiguous (block 1's cols {2,3} come before block 0's {0,1})
    assert set(perm[:2].tolist()) == {2, 3}
    assert set(perm[2:].tolist()) == {0, 1}
    # within block 1, the higher-score column (idx 2, value 10) comes first
    assert perm[0].item() == 2
    # perm is a valid permutation of 0..3
    assert sorted(perm.tolist()) == [0, 1, 2, 3]


def test_compute_group_importance_permutation_variant_max():
    x = torch.tensor([1.0, 5.0, 4.0, 4.0])  # block0 max=5, block1 max=4
    perm, block_order = compute_group_importance_permutation(x, group_size=2, variant="max")
    assert block_order.tolist() == [0, 1]


def test_compute_group_importance_permutation_invalid():
    with pytest.raises(ValueError):
        compute_group_importance_permutation(torch.zeros(4), group_size=0)
    with pytest.raises(ValueError):
        compute_group_importance_permutation(torch.zeros(5), group_size=2)  # not divisible
    with pytest.raises(ValueError):
        compute_group_importance_permutation(torch.zeros(4), group_size=2, variant="bad")


# -----------------------------------------------------------------------------
# Grid helpers read the quantizer through proxy accessors: the result must match
# the previous quant_injector-based derivation (the injector is the ground truth).
# -----------------------------------------------------------------------------


class _IntPerGroupWeight(MXInt8Weight):
    # Groupwise int weight quant with a parameter-from-stats scale (PiSO's
    # supported scaling) instead of MX's power-of-two scale.
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    group_size = 4


class _FloatPerChannelWeight(Fp8e4m3OCPWeightPerChannelFloat):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS


def _build_quant_linear(weight_quant, in_f=8, out_f=6, **kwargs) -> qnn.QuantLinear:
    torch.manual_seed(0)
    layer = qnn.QuantLinear(in_f, out_f, bias=False, weight_quant=weight_quant, **kwargs)
    layer.eval()
    # A forward pass initializes the quantizer parameters.
    with torch.no_grad():
        layer(torch.randn(4, in_f))
    return layer


def test_is_float_weight_quant():
    assert _is_float_weight_quant(_build_quant_linear(_FloatPerChannelWeight).weight_quant) is True
    assert _is_float_weight_quant(
        _build_quant_linear(Int8WeightPerChannelFloatMSE).weight_quant) is False
    assert _is_float_weight_quant(
        _build_quant_linear(_IntPerGroupWeight, weight_group_size=4).weight_quant) is False


def test_is_int_weight_quant():
    assert _is_int_weight_quant(
        _build_quant_linear(Int8WeightPerChannelFloatMSE).weight_quant) is True
    assert _is_int_weight_quant(
        _build_quant_linear(_IntPerGroupWeight, weight_group_size=4).weight_quant) is True
    assert _is_int_weight_quant(_build_quant_linear(_FloatPerChannelWeight).weight_quant) is False


def test_is_int_and_float_weight_quant_are_mutually_exclusive():
    for weight_quant, kwargs in [(Int8WeightPerChannelFloatMSE, {}),
                                 (_IntPerGroupWeight, dict(weight_group_size=4)),
                                 (_FloatPerChannelWeight, {})]:
        wq = _build_quant_linear(weight_quant, **kwargs).weight_quant
        assert _is_int_weight_quant(wq) != _is_float_weight_quant(wq)


def test_proxy_restrict_scale_positive_matches_injector():
    for weight_quant, kwargs in [(Int8WeightPerChannelFloatMSE, {}),
                                 (_IntPerGroupWeight, dict(weight_group_size=4)),
                                 (_FloatPerChannelWeight, {})]:
        wq = _build_quant_linear(weight_quant, **kwargs).weight_quant
        assert wq.restrict_scale_positive == wq.quant_injector.restrict_scale_positive


@pytest.mark.parametrize(
    "weight_quant,kwargs,dtype",
    [(Int8WeightPerChannelFloatMSE, {}, torch.float32),
     (_IntPerGroupWeight, dict(weight_group_size=4), torch.float32),
     (_FloatPerChannelWeight, {}, torch.float64)])
def test_build_grid_matches_injector(weight_quant, kwargs, dtype):
    layer = _build_quant_linear(weight_quant, **kwargs)
    qi = layer.weight_quant.quant_injector
    device = torch.device('cpu')

    grid = _build_grid(layer, dtype=dtype, device=device)

    # Reference grid built directly from the injector (the previous behavior).
    if qi.tensor_quant.__class__.__name__ == 'FloatQuant':
        expected = make_fp_grid(
            e_bits=qi.exponent_bit_width,
            m_bits=qi.mantissa_bit_width,
            exponent_bias=qi.exponent_bias,
            signed=qi.signed)
    else:
        expected = make_int_grid(
            bit_width=qi.bit_width,
            signed=qi.signed,
            narrow_range=layer.weight_quant.is_narrow_range)

    assert grid.dtype == dtype
    assert grid.device == device
    assert grid.tolist() == pytest.approx(expected)


def test_grid_threshold_int_vs_float():
    int_layer = _build_quant_linear(Int8WeightPerChannelFloatMSE)
    grid = _build_grid(int_layer, torch.float32, torch.device('cpu'))
    # int: threshold is the largest magnitude of the (asymmetric) grid.
    assert _grid_threshold(grid, int_layer.weight_quant) == grid.abs().max()

    float_layer = _build_quant_linear(_FloatPerChannelWeight)
    fgrid = _build_grid(float_layer, torch.float32, torch.device('cpu'))
    # float: threshold is the max representable value (grid top).
    assert _grid_threshold(fgrid, float_layer.weight_quant) == fgrid[-1]


# -----------------------------------------------------------------------------
# Stored-scale accessors (_has_base_scale / _get_base_scale / _set_base_scale).
# These take the quant layer and unpack weight_quant.tensor_quant.scaling_impl
# themselves. _get_base_scale returns None (and _set_base_scale raises) unless
# the scale is a ParameterFromStatsFromParameterScaling.
# -----------------------------------------------------------------------------


def _scaling_value(layer):
    """The Parameter the accessors are expected to read and write."""
    return layer.weight_quant.tensor_quant.scaling_impl.value


def test_has_base_scale():
    assert _has_base_scale(_build_quant_linear(Int8WeightPerChannelFloatMSE)) is True
    assert _has_base_scale(_build_quant_linear(_IntPerGroupWeight, weight_group_size=4)) is True
    assert _has_base_scale(_build_quant_linear(_FloatPerChannelWeight)) is True
    # a plain stats-from-parameter scaling has no stored value parameter.
    assert _has_base_scale(_build_quant_linear(Int8WeightPerChannelFloat)) is False


def test_get_base_scale_returns_scaling_parameter():
    layer = _build_quant_linear(Int8WeightPerChannelFloatMSE)
    # the raw stored parameter (identity), not a fresh tensor.
    assert _get_base_scale(layer) is _scaling_value(layer)


def test_get_base_scale_none_without_parameter_from_stats_scale():
    assert _get_base_scale(_build_quant_linear(Int8WeightPerChannelFloat)) is None


def test_get_base_scale_none_when_quant_disabled():
    layer = _build_quant_linear(Int8WeightPerChannelFloatMSE)
    layer.weight_quant.disable_quant = True
    assert _get_base_scale(layer) is None
    assert _has_base_scale(layer) is False


def test_set_base_scale_full_write():
    layer = _build_quant_linear(Int8WeightPerChannelFloatMSE)
    new = torch.full_like(_get_base_scale(layer), 0.5)
    _set_base_scale(layer, new)
    assert torch.equal(_get_base_scale(layer), new)
    # still the same registered Parameter
    assert _get_base_scale(layer) is _scaling_value(layer)


def test_set_base_scale_group_column_write():
    layer = _build_quant_linear(_IntPerGroupWeight, weight_group_size=4)
    before = _get_base_scale(layer).clone()
    # write only group column 1, leave column 0 untouched
    col = torch.full_like(before[:, 1], 9.0)
    _set_base_scale(layer, col, index=(slice(None), 1))
    assert torch.all(_get_base_scale(layer)[:, 1] == 9.0)
    assert torch.equal(_get_base_scale(layer)[:, 0], before[:, 0])


def test_set_base_scale_raises_without_parameter_from_stats_scale():
    layer = _build_quant_linear(Int8WeightPerChannelFloat)
    with pytest.raises(RuntimeError, match="Cannot set base scale"):
        _set_base_scale(layer, torch.zeros(6, 1))


def test_set_base_scale_does_not_require_grad():
    # the write runs under no_grad, so it must not make the Parameter a graph leaf
    layer = _build_quant_linear(Int8WeightPerChannelFloatMSE)
    _set_base_scale(layer, torch.full_like(_get_base_scale(layer), 0.25))
    assert _get_base_scale(layer).grad_fn is None


def test_has_zero_zero_point():
    assert _has_zero_zero_point(_build_quant_linear(Int8WeightPerChannelFloatMSE)) is True
    assert _has_zero_zero_point(
        _build_quant_linear(_IntPerGroupWeight, weight_group_size=4)) is True
