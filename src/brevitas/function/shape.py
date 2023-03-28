# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
    'over_batch_over_output_channels']


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
