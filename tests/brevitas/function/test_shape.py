# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given

from brevitas.function.shape import *
from tests.brevitas.hyp_helper import empty_tensor_random_shape_st


@given(x=empty_tensor_random_shape_st(min_dims=1, max_dims=10))
def test_shape_over_tensor(x):
    """
    Test over_tensor function on an empty tensor of random shape and number of dimensions.
    """
    shape = over_tensor(x)
    assert shape == -1


@given(x=empty_tensor_random_shape_st(min_dims=2, max_dims=10))
def test_shape_over_batch_over_tensor(x):
    """
    Test over_batch_over_tensor function on an empty tensor of random shape and number of dimensions.
    """
    shape = over_batch_over_tensor(x)
    assert len(shape) == 2
    assert shape[0] == x.shape[0]
    assert shape[1] == -1


@given(x=empty_tensor_random_shape_st(min_dims=2, max_dims=10))
def test_shape_over_output_channels(x):
    """
    Test over_output_channels function on an empty tensor of random shape and number of dimensions.
    """
    shape = over_output_channels(x)
    assert len(shape) == 2
    assert shape[0] == x.shape[0]
    assert shape[1] == -1


@given(x=empty_tensor_random_shape_st(min_dims=2, max_dims=10))
def test_shape_over_batch_over_output_channels(x):
    """
    Test over_batch_over_output_channels function on an empty tensor of random shape and number of dimensions.
    """
    shape = over_batch_over_output_channels(x)
    assert len(shape) == 3
    assert shape[0] == x.shape[0]
    assert shape[1] == x.shape[1]
    assert shape[2] == -1
