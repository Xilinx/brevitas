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