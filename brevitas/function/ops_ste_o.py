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


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return round_ste_fn.apply(x)


def tensor_clamp_ste(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    return tensor_clamp_ste_fn.apply(x, min_val, max_val)


def scalar_clamp_ste(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return scalar_clamp_ste_fn.apply(x, min_val, max_val)


def binary_sign_ste(x: torch.Tensor) -> torch.Tensor:
    return binary_sign_ste_fn.apply(x)


def ternary_sign_ste(x: torch.Tensor) -> torch.Tensor:

    return ternary_sign_ste_fn.apply(x)


def max_uint(narrow_range: bool, bit_width: torch.Tensor):
    if narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** bit_width) - 1
    value = round_ste(value)
    return value


def max_int(signed: bool, bit_width: torch.Tensor):
    if signed:
        value = (2 ** (bit_width - 1)) - 1
    else:
        value = (2 ** bit_width) - 1
    value = round_ste(value)
    return value


def min_int(signed: bool, narrow_range: bool, bit_width: torch.Tensor):
    if signed and narrow_range:
        value = - (2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = - (2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    value = round_ste(value)
    return value