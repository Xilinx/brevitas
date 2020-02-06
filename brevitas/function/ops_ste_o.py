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
    """ Perform round operation with Straight Trough Estimation (STE) of the Gradient

    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.


    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the round operation

    Returns
    -------
    Tensor
        Tensor after applying round operation.
        When backpropagating through this value, a straight through estimator is applied.

    """
    return round_ste_fn.apply(x)


def ceil_ste(x: torch.Tensor) -> torch.Tensor:
    """ Perform ceil operation with Straight Trough Estimation (STE) of the Gradient

    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.

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

    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.

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


def tensor_clamp_ste(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """ Perform tensor-clamp operation with Straight Trough Estimation (STE) of the Gradient

    This function accepts two Tensors as `min_val` and `max_val`. These Tensors must have the same shape as
    `x`, so that each element of `x` can be clamped according to the correspondent min_val and max_val.
    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.


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
        When backpropagating through this value, a straight through estimator is applied.
    """
    return tensor_clamp_ste_fn.apply(x, min_val, max_val)


def scalar_clamp_ste(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """ Perform clamp operation with Straight Trough Estimation (STE) of the Gradient

    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.


    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the clamp operation
    min_val : Float
        Scalar containing the minimum value for the clamp operation
    max_val : Float
        Scalar containing the maximum value for the clamp operation

    Returns
    -------
    Tensor
        Tensor for which every element of `x` is clamped between `min_val` and `max_val`.
        When backpropagating through this value, a straight through estimator is applied.
    """
    return scalar_clamp_ste_fn.apply(x, min_val, max_val)


def binary_sign_ste(x: torch.Tensor) -> torch.Tensor:
    """ Perform binarization with Straight Trough Estimation (STE) of the Gradient

    This operation performs binarization on the input Tensor.
    The output value will be one for each input value >= 0, otherwise it will be 0.
    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.


    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the binarization operation

    Returns
    -------
    Tensor
        Tensor after applying binarization.
        When backpropagating through this value, a straight through estimator is applied.

    """
    return binary_sign_ste_fn.apply(x)


def ternary_sign_ste(x: torch.Tensor) -> torch.Tensor:
    """ Perform ternary operator with Straight Trough Estimation (STE) of the Gradient

    This operations behaves as the function `sign` of Pytorch.
    This operation behaves like an identity on the backward pass.
    For Pytorch version >= 1.3.0, the STE operator is implemented in C++ using the
    torch::autograd::Function class and compiled. At execution time, the Just-In-Time (JIT) compiler of Pytorch
    is used to speed-up the computation.
    For Pytorch version < 1.3.0, the STE operator is implemented using the
    torch.autograd.Function class in python, and the JIT cannot be used.


    Parameters
    ----------
    x : Tensor
        Tensor on which to apply the ternary operation

    Returns
    -------
    Tensor
        Tensor after applying ternary operation.
        When backpropagating through this value, a straight through estimator is applied.

    """
    return ternary_sign_ste_fn.apply(x)
