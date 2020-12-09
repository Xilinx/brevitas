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
Implementation of various functions with a straight-through gradient estimator, dispatched to
either a native just-in-time compiled backend (when env BREVITAS_JIT=1) or to a torch.autograd.Function
implemented in :obj:`~brevitas.function.autograd_ste_ops` (when env BREVITAS_JIT=0).
The native backend is enabled when BREVITAS_JIT is enabled to allow for end-to-end compilation of
the built-in quantizers, since as of Pytorch 1.7.0 a torch.autograd.Function is not supported by the compiler.
"""

import torch
from torch import Tensor

import brevitas

__all__ = [
    'round_ste',
    'ceil_ste',
    'floor_ste',
    'tensor_clamp_ste',
    'scalar_clamp_ste',
    'scalar_clamp_min_ste',
    'binary_sign_ste',
    'ternary_sign_ste',
    'round_to_zero_ste',
    'abs_binary_sign_grad'
]


if brevitas.NATIVE_STE_BACKEND_LOADED:
    fn_prefix = torch.ops.autograd_ste_ops
    script_flag = brevitas.jit.script
else:
    from brevitas.function import autograd_ste_ops as fn_prefix
    script_flag = torch.jit.ignore


@script_flag
def round_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.round_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.round_ste_impl(x)


@script_flag
def ceil_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.ceil_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.ceil_ste_impl(x)


@script_flag
def floor_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.floor_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.floor_ste_impl(x)


@script_flag
def tensor_clamp_ste(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.tensor_clamp_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    output = fn_prefix.tensor_clamp_ste_impl(x, min_val, max_val)
    return output


@script_flag
def scalar_clamp_ste(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.scalar_clamp_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.scalar_clamp_ste_impl(x, min_val, max_val)


@script_flag
def scalar_clamp_min_ste(x: Tensor, min_val: float) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.scalar_clamp_min_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.scalar_clamp_min_ste_impl(x, min_val)


@script_flag
def binary_sign_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.binary_sign_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.binary_sign_ste_impl(x)


@script_flag
def ternary_sign_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.ternary_sign_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.ternary_sign_ste_impl(x)


@script_flag
def round_to_zero_ste(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.round_to_zero_ste_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.round_to_zero_ste_impl(x)


@script_flag
def abs_binary_sign_grad(x: Tensor) -> Tensor:
    """
    Wrapper for either :func:`~brevitas.function.autograd_ste_ops.abs_binary_sign_grad_impl` (with env
    BREVITAS_JIT=0) or its native just-in-time compiled variant (with BREVITAS_JIT=1).
    """
    return fn_prefix.abs_binary_sign_grad_impl(x)