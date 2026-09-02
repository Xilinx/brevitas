# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest
import torch

from brevitas.utils.quant_utils import groupwise_dequant_expand
from brevitas.utils.torch_utils import padding_to_multiple
from tests.marker import jit_disabled_for_compile
from tests.marker import requires_pt_ge
from tests.marker import requires_torch_compile


@pytest.mark.parametrize(
    'shape, dim, multiple, expected_shape',
    [((2, 5), 1, 4, (2, 8)), ((2, 8), 1, 4, (2, 8)), ((2, 3, 5), -1, 4, (2, 3, 8)),
     ((5, 3), 0, 4, (8, 3))])
@requires_pt_ge('2.2')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_padding_to_multiple(shape, dim, multiple, expected_shape):
    x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    compiled_fn = torch.compile(padding_to_multiple, backend='eager', fullgraph=True)

    actual = compiled_fn(x, dim, multiple)
    expected = padding_to_multiple(x, dim, multiple)

    assert actual.shape == expected_shape
    assert torch.equal(actual, expected)
    assert expected.is_contiguous()
    assert torch.equal(torch.narrow(actual, dim, 0, shape[dim]), x)

    padding = expected_shape[dim] - shape[dim]
    if padding:
        padded_values = torch.narrow(actual, dim, shape[dim], padding)
        assert torch.count_nonzero(padded_values) == 0


@pytest.mark.parametrize('group_dim', [1, -1])
@pytest.mark.parametrize('scalar_metadata', [False, True])
@pytest.mark.parametrize('expand_metadata', [False, True])
@requires_pt_ge('2.2')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_groupwise_dequant_expand(group_dim, scalar_metadata, expand_metadata):
    value = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    if scalar_metadata:
        scale = torch.tensor(2.)
        zero_point = torch.tensor(1.)
    else:
        scale = torch.tensor([1., 2., 3., 4.]).reshape(2, 2, 1)
        zero_point = torch.tensor([10., 20., 30., 40.]).reshape(2, 2, 1)
    dequant_shape = (2, 5)
    compiled_fn = torch.compile(groupwise_dequant_expand, backend='eager', fullgraph=True)

    actual = compiled_fn(value, scale, zero_point, group_dim, dequant_shape, expand_metadata)
    expected = groupwise_dequant_expand(
        value, scale, zero_point, group_dim, dequant_shape, expand_metadata)

    assert all(
        torch.equal(actual_value, expected_value) for actual_value,
        expected_value in zip(actual, expected))
    assert torch.equal(actual[0], torch.tensor([[0., 1., 2., 3., 4.], [8., 9., 10., 11., 12.]]))

    if not expand_metadata or scalar_metadata:
        assert actual[1].shape == scale.shape
        assert actual[2].shape == zero_point.shape
    else:
        assert torch.equal(actual[1], torch.tensor([[1., 1., 1., 1., 2.], [3., 3., 3., 3., 4.]]))
        assert torch.equal(
            actual[2], torch.tensor([[10., 10., 10., 10., 20.], [30., 30., 30., 30., 40.]]))
