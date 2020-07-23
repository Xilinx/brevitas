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

import torch

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
def tensor_clamp_(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    torch.min(x, max_val, out=x)
    torch.max(x, min_val, out=x)
    return x


@torch.jit.script
def identity(x: torch.Tensor) -> torch.Tensor:
    """ Identity function

    Parameters
    ----------
    x : Tensor
        Input Tensor

    Returns
    -------
    Tensor
        Unaltered input tensor

    """
    return x


@torch.jit.script
def max_uint(narrow_range: bool, bit_width: torch.Tensor) -> torch.Tensor:
    """ Compute the maximum unsigned integer representable

    The maximum unsigned integer representable depends on the number of bits, and whether the narrow range setting
    is used. If so, the maximum value represented is decreased by one unit.

    Parameters
    ----------
    narrow_range : Bool
        Flag that indicates whether to decrease the possible maximum value represented
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Maximum unsigned integer that can be represented according to the input parameters

    """
    if narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** bit_width) - 1
    return value


@torch.jit.script
def max_int(signed: bool, bit_width: torch.Tensor) -> torch.Tensor:
    """ Compute the maximum integer representable

    The maximum integer representable depends on the number of bits, and whether the negative numbers are included
    in the representation. If so, one bit is lost in the computation of the maximum value.

    Parameters
    ----------
    signed : Bool
        Flag that indicates whether negative numbers must be included or not
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Maximum integer that can be represented according to the input parameters

    """
    if signed:
        value = (2 ** (bit_width - 1)) - 1
    else:
        value = (2 ** bit_width) - 1
    return value


@torch.jit.script
def min_int(signed: bool, narrow_range: bool, bit_width: torch.Tensor) -> torch.Tensor:
    """ Compute the minimum integer representable

    The minimum integer representable depends on the number of bits, whether the negative numbers are included
    in the representation, and whether the narrow range setting is used.
    For positive-only number, the minimum value will always be zero.
    If the sign and narrow range flags are both set, then the representation will be such that there is symmetry
    between positive and negative values.
    For example, for 3 bit representation, with sign and narrow range, the
    values representable are in the range [-3, 3].
    If the narrow range is not enabled, then the possible values will be in the range [-4, 3].

    Parameters
    ----------
    signed : Bool
        Flag that indicates whether negative numbers must be included or not
    narrow_range : Bool
        Flag that indicates whether the narrow range setting is enabled or not
    bit_width : Tensor
        Number of bits available for the representation

    Returns
    -------
    Tensor
        Minimum integer that can be represented according to the input parameters

    """
    if signed and narrow_range:
        value = - (2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = - (2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value
