# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementation of various core operations often performed as part of quantization.
The implemented functions adheres to the restriction imposed by Pytorch 1.1.0's TorchScript compiler.
"""

from typing import Union

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
    y = positive_mask.to(x.dtype) * 2. - 1.
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
def max_int(
        signed: Union[bool, Tensor], narrow_range: Union[bool, Tensor],
        bit_width: Tensor) -> Tensor:
    """ Compute the maximum integer representable by a given number of bits.

    Args:
        signed (Union[bool, Tensor]): Indicates whether the represented integer is signed or not.
        narrow_range (Union[bool, Tensor]): Indicates whether to narrow the maximum unsigned value represented by 1.
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
    # Workaround: required for compatibility with the JIT for PT=2.2.2
    _signed = bool(signed.item()) if isinstance(signed, Tensor) else signed
    _narrow_range = bool(narrow_range.item()) if isinstance(narrow_range, Tensor) else narrow_range
    if not _signed and not _narrow_range:
        value = (2 ** bit_width) - 1
    elif not _signed and _narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value


@brevitas.jit.script
def min_int(
        signed: Union[bool, Tensor], narrow_range: Union[bool, Tensor],
        bit_width: Tensor) -> Tensor:
    """ Compute the minimum integer representable by a given number of bits.

    Args:
        signed (Union[bool, Tensor]): Indicates whether the represented integer is signed or not.
        narrow_range (Union[bool, Tensor]): Indicates whether to narrow the minimum value represented by 1.
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
    # Workaround: required for compatibility with the JIT for PT=2.2.2
    _signed = bool(signed.item()) if isinstance(signed, Tensor) else signed
    _narrow_range = bool(narrow_range.item()) if isinstance(narrow_range, Tensor) else narrow_range
    if _signed and _narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif _signed and not _narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value


def compute_max_mantissa(mantissa_bit_width: Tensor) -> Tensor:
    # `.data` is needed for avoind errors during gradient computation.
    # TODO: Check if/how `.data` affects gradient propagation and accuracy
    return torch.sum((2. ** torch.arange(0, -1. * mantissa_bit_width.data - 1., -1.)))


@brevitas.jit.ignore
def max_float(exponent_bit_width: Tensor, max_mantissa: Tensor, exponent_bias: Tensor):
    max_exponent = (2. ** exponent_bit_width) - 1. - exponent_bias
    max_val = max_mantissa * (2 ** max_exponent)
    return max_val


def get_upper_bound_on_l1_norm(
        accumulator_bit_width: Tensor, input_bit_width: Tensor, input_is_signed: bool) -> Tensor:
    """Calculate the upper bound on the l1-norm of the weights needed to guarantee overflow avoidance
    for a given accumulator bit width and input representation using the derivations from
    `A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance` by I.Colbert,
    A.Pappalardo, and J.Petri-Koenig. Note that this assumes integer quantization."""
    assert input_bit_width is not None, "A2Q relies on input bit-width."
    assert input_is_signed is not None, "A2Q relies on input sign."
    assert accumulator_bit_width is not None, "A2Q relies on accumulator bit-width."
    input_is_signed = float(input_is_signed)  # 1. if signed else 0.
    max_accumulator_bit_width = accumulator_bit_width  # P
    max_accumulator_mag = pow(2., max_accumulator_bit_width - 1.) - 1.  # 2^{P-1}-1
    max_input_mag_inverse = pow(2., input_is_signed - input_bit_width)
    return max_accumulator_mag * max_input_mag_inverse
