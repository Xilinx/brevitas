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

"""
Implementation of various functions to compute shapes that induce flattening along certain
dimensions of a tensor.
"""

from typing import Tuple

import torch
from torch import Tensor

import brevitas

__all__ = [
    'over_tensor',
    'over_output_channels',
    'over_batch_over_tensor',
    'over_batch_over_output_channels'
]


@brevitas.jit.script
def over_tensor(x: Tensor) -> int:
    """
    Computes the shape s such that x.view(s) is a flat tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        The number -1 corresponding to a flat shape.

    Examples:
        >>> over_tensor(torch.randn([2, 3, 4, 3]))
        -1
    """
    return -1


@brevitas.jit.script
def over_output_channels(x: Tensor) -> Tuple[int, int]:
    """
    Computes the shape s such that x.view(s) is a 2-dim tensor with output channels
    at dimension 0 and any other feature at dimension 1.

    Args:
    x (Tensor): Input tensor with output channels at dimension 0.

    Returns:
        A tuple containing the 2-dim shape.

    Examples:
        >>> over_output_channels(torch.randn([2, 3, 4, 3]))
        (2, -1)
    """
    return x.shape[0], -1


@brevitas.jit.script
def over_batch_over_tensor(x: Tensor) -> Tuple[int, int]:
    """
    Computes the shape s such that x.view(s) is a 2-dim tensor with batches
    at dimension 0 and any other feature at dimension 1.

    Args:
        x (Tensor): Input tensor with batches at dimension 0.

    Returns:
        A tuple containing the 2-dim shape.

    Examples:
        >>> over_batch_over_tensor(torch.randn([2, 3, 4, 3]))
        (2, -1)
    """
    return x.shape[0], -1


@brevitas.jit.script
def over_batch_over_output_channels(x: Tensor):
    """
    Returns a shape s such that x.view(s) is a 3-dim tensor with batches
    at dimension 0, output channels at dimension 1, and any other feature at dimension 2.

    Args:
        x (Tensor): Input tensor with batches at dimension 0 and output channels at dimension 1.

    Returns:
        A tuple containing the 3-dim shape.

    Examples:
        >>> over_batch_over_output_channels(torch.randn([2, 3, 4, 3]))
        (2, 3, -1)
    """
    return x.shape[0], x.shape[1], -1