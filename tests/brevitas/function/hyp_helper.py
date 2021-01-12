# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import hypothesis.strategies as st

import torch

from tests.brevitas.hyp_helper import float_st, float_tensor_nz_st
from tests.brevitas.hyp_helper import float_tensor_st, min_max_tensor_st, random_tensor_shape_st
from tests.brevitas.hyp_helper import min_max_scalar_tensor_st, float_tensor_random_shape_st


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
