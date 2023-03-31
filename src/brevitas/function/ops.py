# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementation of various core operations often performed as part of quantization.
The implemented functions adheres to the restriction imposed by Pytorch 1.1.0's TorchScript compiler.
"""

import torch
from torch import Tensor

import brevitas


@brevitas.jit.script
def binary_sign(x: Tensor) -> Tensor:
    """
    Computes the 2-valued sign of an input tensor.

    Args:
        x (Tensor): input tensor.

    Returns:
        Tensor: the 2-valued sign tensor of the input tensor.

    Examples:
        >>> binary_sign(torch.tensor([2.1, -0.3, 0.0]))
        tensor([ 1., -1.,  1.])
    """
    positive_mask = torch.ge(x, 0.0)
    negative_mask = torch.lt(x, 0.0)
    y = positive_mask.to(x.dtype) - negative_mask.to(x.dtype)
    return y


@brevitas.jit.script
def round_to_zero(x: Tensor) -> Tensor:
    """
    Compute rounding towards zero.

    Args:
        x (Tensor): input tensor.

    Returns:
        Tensor: rounded input tensor.

    Examples:
        >>> round_to_zero(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        tensor([-1., -0.,  0.,  1.])
    """
    y = torch.sign(x) * torch.floor(torch.abs(x))
    return y


@brevitas.jit.script
def dpu_round(x: Tensor) -> Tensor:
    """
    Compute DPU rounding.

    Args:
        x (Tensor): input tensor.

    Returns:
        Tensor: rounded input tensor.

    Examples:
        >>> dpu_round(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        tensor([-1., -0.,  0.,  2.])
    """
    y = torch.where((x < 0.) & (x - torch.floor(x) == 0.5), torch.ceil(x), torch.round(x))
    return y


@brevitas.jit.script
def tensor_clamp(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    Generalized clamp function with support for tensors as clamping values.

    Args:
        x: Input on which to apply the clamp operation
        min_val: Minimum values for the clamp operation.
        max_val: Maximum values for the clamp operation.

    Notes:
        x, min_val, max_val need to be broadcastable.

    Notes:
        Differentiable w.r.t. x, min_val, max_val.

    Returns:
        Input `x` clamped between the provided minimum and maximum tensors.

    Examples:
        >>> tensor_clamp(torch.tensor([1.7, -0.5, 0.1]), torch.tensor(0.0), torch.tensor(1.0))
        tensor([1.0000, 0.0000, 0.1000])
    """
    out = torch.where(x > max_val, max_val.type_as(x), x)
    out = torch.where(out < min_val, min_val.type_as(out), out)
    return out


@brevitas.jit.script
def tensor_clamp_(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    In-place variant of :func:`~brevitas.function.ops.tensor_clamp`.
    Not differentiable wrt to any of the inputs.
    """
    torch.min(x, max_val, out=x)
    torch.max(x, min_val, out=x)
    return x


@brevitas.jit.script
def identity(x: Tensor) -> Tensor:
    """
    Identity function.

    Args:
        x (Tensor): Input Tensor

    Returns:
        Tensor: THe input tensor x

    Examples:
        >>> identity(torch.tensor(1.7))
        tensor(1.7)
    """
    return x


@brevitas.jit.script
def max_int(signed: bool, narrow_range: bool, bit_width: Tensor) -> Tensor:
    """ Compute the maximum integer representable by a given number of bits.

    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the maximum unsigned value represented by 1.
        bit_width (Tensor): Number of bits available for the representation.

    Returns:
        Tensor: Maximum integer that can be represented according to the input arguments.

    Examples:
        >>> max_int(signed=True, narrow_range=True, bit_width=torch.tensor(8))
        tensor(127)
        >>> max_int(signed=False, narrow_range=True, bit_width=torch.tensor(8))
        tensor(254)
        >>> max_int(signed=True, narrow_range=False, bit_width=torch.tensor(8))
        tensor(127)
        >>> max_int(signed=False, narrow_range=False, bit_width=torch.tensor(8))
        tensor(255)
    """
    if not signed and not narrow_range:
        value = (2 ** bit_width) - 1
    elif not signed and narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value


@brevitas.jit.script
def min_int(signed: bool, narrow_range: bool, bit_width: Tensor) -> Tensor:
    """ Compute the minimum integer representable by a given number of bits.

    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the minimum value represented by 1.
        bit_width (Tensor): Number of bits available for the representation.

    Returns:
        Tensor: Maximum unsigned integer that can be represented according to the input arguments.

    Examples:
        >>> min_int(signed=True, narrow_range=True, bit_width=torch.tensor(8))
        tensor(-127)
        >>> min_int(signed=False, narrow_range=True, bit_width=torch.tensor(8))
        tensor(0)
        >>> min_int(signed=True, narrow_range=False, bit_width=torch.tensor(8))
        tensor(-128)
        >>> min_int(signed=False, narrow_range=False, bit_width=torch.tensor(8))
        tensor(0)
    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value
