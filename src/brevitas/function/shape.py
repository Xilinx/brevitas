from typing import Tuple
from torch import Tensor

import brevitas


@brevitas.jit.script
def over_tensor(x: Tensor) -> int:
    """
    Computes the shape s such that x.view(s) is a flat tensor

    Args:
        x (Tensor): Input tensor

    Returns:
        The number -1 corresponding to a flat shape
    """
    return -1


@brevitas.jit.script
def over_output_channels(x: Tensor) -> Tuple[int, int]:
    """
    Computes the shape s such that x.view(s) is a 2-dim tensor with output channels
    at dimension 0 and any other feature at dimension 1.

    Args:
    x (Tensor): Input tensor with output channels at dimension 0

    Returns:
        A tuple containing the 2-dim shape
    """
    return x.shape[0], -1


@brevitas.jit.script
def over_batch_over_tensor(x: Tensor) -> Tuple[int, int]:
    """
    Computes the shape s such that x.view(s) is a 2-dim tensor with batches
    at dimension 0 and any other feature at dimension 1.

    Args:
        x (Tensor): Input tensor with batches at dimension 0

    Returns:
        A tuple containing the 2-dim shape
    """
    return x.shape[0], -1


@brevitas.jit.script
def over_batch_over_output_channels(x: Tensor):
    """
    Returns a shape s such that x.view(s) is a 3-dim tensor with batches
    at dimension 0, output channels at dimension 1, and any other feature at dimension 2.

    Args:
        x (Tensor): Input tensor with batches at dimension 0 and output channels at dimension 1

    Returns:
        A tuple containing the 3-dim shape
    """
    return x.shape[0], x.shape[1], -1