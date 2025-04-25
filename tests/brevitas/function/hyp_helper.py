# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hypothesis.strategies as st
import torch

from tests.brevitas.hyp_helper import float_st
from tests.brevitas.hyp_helper import float_tensor_nz_st
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import float_tensor_st
from tests.brevitas.hyp_helper import min_max_scalar_tensor_st
from tests.brevitas.hyp_helper import min_max_tensor_st
from tests.brevitas.hyp_helper import random_tensor_shape_st


def tensor_clamp_test_st():
    """
    Generate input values for testing tensor_clamp.
    """
    return st.one_of(
        tensor_clamp_random_shape_test_st(), tensor_clamp_min_max_scalar_tensor_test_st())


def binary_sign_test_st():
    """
    Generate either zero or float tensors of random shape.
    """
    return st.one_of(st.just(torch.tensor(0.0)), float_tensor_random_shape_st())


def tensor_clamp_ste_test_st():
    """
    Generate input values for testing tensor_clamp_ste fwd or bwd.
    """
    return st.one_of(
        tensor_clamp_ste_min_max_scalar_tensor_test_st(), tensor_clamp_ste_random_shape_test_st())


@st.composite
def scalar_clamp_ste_test_st(draw):
    """
    Generate min_val and max_val floats, val and val_grad tensors.
    The val and val_grad tensors has the same random shape
    """
    shape = draw(random_tensor_shape_st())
    min_val = draw(float_st())
    max_val = draw(float_st())
    val = draw(float_tensor_st(shape))
    val_grad = draw(float_tensor_nz_st(shape))
    return min_val, max_val, val, val_grad


@st.composite
def tensor_clamp_random_shape_test_st(draw):
    """
    Generate min_val, max_val and val tensors all of the same random shape.
    """
    shape = draw(random_tensor_shape_st())
    min_val, max_val = draw(min_max_tensor_st(shape))
    val = draw(float_tensor_st(shape))
    return min_val, max_val, val


@st.composite
def tensor_clamp_min_max_scalar_tensor_test_st(draw):
    """
    Generate min_val, max_val and val tensors.
    The val tensor has random shape, min_val and max_val are scalar tensors.
    """
    shape = draw(random_tensor_shape_st())
    min_val, max_val = draw(min_max_scalar_tensor_st())
    val = draw(float_tensor_st(shape))
    return min_val, max_val, val


@st.composite
def tensor_clamp_ste_random_shape_test_st(draw):
    """
    Generate min_val, max_val, val and val_grad tensors all of the same random shape.
    """
    shape = draw(random_tensor_shape_st())
    min_val, max_val = draw(min_max_tensor_st(shape))
    val = draw(float_tensor_st(shape))
    val_grad = draw(float_tensor_nz_st(shape))
    return min_val, max_val, val, val_grad


@st.composite
def tensor_clamp_ste_min_max_scalar_tensor_test_st(draw):
    """
    Generate min_val, max_val, val and val_grad tensors.
    The val and val_grad tensors has the same random shape, min_val and max_val are scalar tensors.
    """
    shape = draw(random_tensor_shape_st())
    min_val, max_val = draw(min_max_scalar_tensor_st())
    val = draw(float_tensor_st(shape))
    val_grad = draw(float_tensor_nz_st(shape))
    return min_val, max_val, val, val_grad


@st.composite
def scalar_clamp_min_ste_test_st(draw):
    """
    Generate min_val float, val and val_grad tensors.
    The val and val_grad tensors has the same random shape.
    """
    shape = draw(random_tensor_shape_st())
    min_val = draw(float_st())
    val = draw(float_tensor_st(shape))
    val_grad = draw(float_tensor_st(shape))
    return min_val, val, val_grad
