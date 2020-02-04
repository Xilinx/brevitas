# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
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

from brevitas.function.autograd_ops import *


@torch.jit.script
def tensor_clamp(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """

    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the clamp operation
    min_val : Tensor
        Tensor containing the minimum values for the clamp operation. Must have the same shape of `x`
    max_val : Tensor
        Tensor containing the maximum values for the clamp operation. Must have the same shape of `x`

    Returns
    -------
    Tensor
        Tensor for which every element of `x` is clamped between the corresponding minimum and maximum values.
    """
    out = torch.where(x > max_val, max_val, x)
    out = torch.where(out < min_val, min_val, out)
    return out


@torch.jit.script
def tensor_clamp_(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
    torch.min(x, max_val, out=x)
    torch.max(x, min_val, out=x)
    return x


def ceil_ste(x: torch.Tensor) -> torch.Tensor:
    """ Perform ceil operation with Straight Trough Estimation (STE) of the Gradient

    This operation behaves like an identity on the backward pass. The STE is implemented using the
    torch.autograd.Function class in python, due to some unexpected behaviour of at::ceil implementation in C++

    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the ceil operation

    Returns
    -------
    Tensor
        Tensor after applying ceil operation. When backpropagating, the gradient will be unaffected by the ceil
        operation

    """
    return ceil_ste_fn.apply(x)


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """ Perform floor operation with Straight Trough Estimation (STE) of the Gradient

    This operation behaves like an identity on the backward pass. The STE is implemented using the
    torch.autograd.Function class in python, due to some unexpected behaviour of at::floor implementation in C++

    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the floor operation

    Returns
    -------
    Tensor
        Tensor after applying floor operation. When backpropagating, the gradient will be unaffected by the floor
        operation

    """
    return floor_ste_fn.apply(x)

