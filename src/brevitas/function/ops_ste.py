# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementation of various functions with a straight-through gradient estimators, dispatched to
either a native just-in-time compiled backend (when env ``BREVITAS_JIT=1``) or to an autograd
Function implemented in :obj:`~brevitas.ops.autograd_ste_ops` (when env ``BREVITAS_JIT=0``).

The native backend is enabled when ``BREVITAS_JIT`` is enabled to allow for end-to-end compilation
of the built-in quantizers, since as of Pytorch 1.8.1 a torch.autograd.Function is not supported by
the compiler.
"""

import torch
from torch import Tensor

import brevitas
from brevitas.function.ops import binary_sign
from brevitas.function.ops import dpu_round
from brevitas.function.ops import round_to_zero
from brevitas.function.ops import tensor_clamp
from brevitas.function.ops import tensor_clamp_

__all__ = [
    'round_ste',
    'ceil_ste',
    'floor_ste',
    'tensor_clamp_ste',
    'tensor_clamp_ste_',
    'scalar_clamp_ste',
    'scalar_clamp_min_ste',
    'binary_sign_ste',
    'ternary_sign_ste',
    'round_to_zero_ste',
    'dpu_round_ste',
    'abs_binary_sign_grad']

if brevitas.NATIVE_STE_BACKEND_LOADED:
    fn_prefix = torch
    script_flag = brevitas.jit.script
else:
    fn_prefix = brevitas
    script_flag = torch.jit.ignore


@script_flag
def round_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`torch.round` with a straight-through gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.round_ste_impl` (with env
        ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = round_ste(x)
        >>> y
        tensor([ 2., -2.], grad_fn=<RoundSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.round(x)
    return fn_prefix.ops.autograd_ste_ops.round_ste_impl(x)


@script_flag
def ceil_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`torch.ceil` with a straight-through gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.ceil_ste_impl` (with env
        ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = ceil_ste(x)
        >>> y
        tensor([ 2., -1.], grad_fn=<CeilSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.ceil(x)
    return fn_prefix.ops.autograd_ste_ops.ceil_ste_impl(x)


@script_flag
def floor_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`torch.floor` with a straight-through gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.floor_ste_impl` (with env
        ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = floor_ste(x)
        >>> y
        tensor([ 1., -2.], grad_fn=<FloorSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.floor(x)
    return fn_prefix.ops.autograd_ste_ops.floor_ste_impl(x)


@script_flag
def tensor_clamp_ste(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    Function that implements :func:`~brevitas.function.ops.tensor_clamp` with a straight-through
    gradient estimator for the gradient of y w.r.t. to x, while the gradient of y w.r.t. to min_val
    and max_val is always None.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.tensor_clamp_ste_impl` (with
        env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = tensor_clamp_ste(x, torch.tensor([-1.0, -0.5, -0.5]), torch.tensor([1.0, 0.5, 0.5]))
        >>> y
        tensor([ 1.0000,  0.4000, -0.5000], grad_fn=<TensorClampSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return tensor_clamp(x, min_val, max_val)
    output = fn_prefix.ops.autograd_ste_ops.tensor_clamp_ste_impl(x, min_val, max_val)
    return output


@script_flag
def tensor_clamp_ste_(x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
    """
    Function that implements :func:`~brevitas.function.ops.tensor_clamp_` with a straight-through
    gradient estimator for the gradient of y w.r.t. to x, while the gradient of y w.r.t. to min_val
    and max_val is always None.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.tensor_clamp_ste_impl_` (with
        env ``BREVITAS_JIT=0``) or its C++ just-in-time compiled variant (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = tensor_clamp_ste_(x, torch.tensor([-1.0, -0.5, -0.5]), torch.tensor([1.0, 0.5, 0.5]))
        >>> y
        tensor([ 1.0000,  0.4000, -0.5000], grad_fn=<InplaceTensorClampSteFnBackward>)
        >>> (y == x).all().item()
        True
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return tensor_clamp_(x, min_val, max_val)
    output = fn_prefix.ops.autograd_ste_ops.tensor_clamp_ste_impl_(x, min_val, max_val)
    return output


@script_flag
def scalar_clamp_ste(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Function that implements :func:`torch.clamp` with a straight-through gradient estimator
    for the gradient of the output w.r.t. to ``x``, while the gradient of ``y`` w.r.t. to ``min_val``
    and ``max_val`` is always ``None``.

    Args:
        x: input tensor to clamp.
        min_val: scalar value to use as lower bound for the input tensor.
        max_val: scalar value to use as upper bound for the input tensor.

    Returns:
        Tensor: clamped output tensor.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.scalar_clamp_ste_impl`
        (with env ``BREVITAS_JIT=0``) or its C++ just-in-time compiled variant
        (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = scalar_clamp_ste(x, -1.0, 1.0)
        >>> y
        tensor([ 1.0000,  0.4000, -1.0000], grad_fn=<ScalarClampSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.clamp(x, min_val, max_val)
    return fn_prefix.ops.autograd_ste_ops.scalar_clamp_ste_impl(x, min_val, max_val)


@script_flag
def scalar_clamp_min_ste(x: Tensor, min_val: float) -> Tensor:
    """
    Function that implements :func:`torch.clamp_min` with a straight-through gradient estimator
    for the gradient of output y w.r.t. to ``x``, while the gradient of y w.r.t. to ``min_val`` is
    always ``None``.

    Args:
        x: input tensor to clamp.
        min_val: scalar value to use as lower bound for the input tensor.

    Returns:
        Tensor: clamped output tensor.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.scalar_clamp_min_ste_impl`
        (with env ``BREVITAS_JIT=0``) or its C++ just-in-time compiled variant
        (with ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = scalar_clamp_min_ste(x, -1.0)
        >>> y
        tensor([ 1.5000,  0.4000, -1.0000], grad_fn=<ScalarClampMinSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.clamp_min(x, min_val)
    return fn_prefix.ops.autograd_ste_ops.scalar_clamp_min_ste_impl(x, min_val)


@script_flag
def binary_sign_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`~brevitas.function.ops.binary_sign` with a straight-through
    gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.binary_sign_ste_impl` (with
        env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, 0.0, -0.5], requires_grad=True)
        >>> y = binary_sign_ste(x)
        >>> y
        tensor([ 1.,  1., -1.], grad_fn=<BinarySignSteFnBackward>)
        >>> grad = torch.tensor([0.1, 0.2, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return binary_sign(x)
    return fn_prefix.ops.autograd_ste_ops.binary_sign_ste_impl(x)


@script_flag
def ternary_sign_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`torch.sign` with a straight-through gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.ternary_sign_ste_impl` (with
        env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, 0.0, -0.5], requires_grad=True)
        >>> y = ternary_sign_ste(x)
        >>> y
        tensor([ 1.,  0., -1.], grad_fn=<TernarySignSteFnBackward>)
        >>> grad = torch.tensor([0.1, 0.2, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.sign(x)
    return fn_prefix.ops.autograd_ste_ops.ternary_sign_ste_impl(x)


@script_flag
def round_to_zero_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`~brevitas.function.ops.round_to_zero` with a straight-through
    gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.round_to_zero_ste_impl` (with
        env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = round_to_zero_ste(x)
        >>> y
        tensor([ 1., -1.], grad_fn=<RoundToZeroSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return round_to_zero(x)
    return fn_prefix.ops.autograd_ste_ops.round_to_zero_ste_impl(x)


@script_flag
def dpu_round_ste(x: Tensor) -> Tensor:
    """
    Function that implements :func:`~brevitas.function.ops.dpu_round` with a straight-through
    gradient estimator.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.dpu_round_ste_impl` (with
        env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = dpu_round_ste(x)
        >>> y
        tensor([ 2., -2.], grad_fn=<DPURoundSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return dpu_round(x)
    return fn_prefix.ops.autograd_ste_ops.dpu_round_ste_impl(x)


@script_flag
def abs_binary_sign_grad(x: Tensor) -> Tensor:
    """
    Function that implements :func:`torch.abs` with a binary-sign backward, in order to
    have subgradient 1 in 0. Compare with :func:`torch.abs`' subgradient of 0 in 0.

    Notes:
        Wrapper for either :func:`~brevitas.ops.autograd_ste_ops.abs_binary_sign_grad_impl`
        (with env ``BREVITAS_JIT=0``) or its native just-in-time compiled variant (with
        ``BREVITAS_JIT=1``).

    Examples:
        >>> x = torch.tensor([0.0], requires_grad=True)
        >>> y = abs_binary_sign_grad(x)
        >>> y
        tensor([0.], grad_fn=<AbsBinarySignGradFnBackward>)
        >>> grad = torch.tensor([0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    if torch._C._get_tracing_state():
        return torch.abs(x)
    return fn_prefix.ops.autograd_ste_ops.abs_binary_sign_grad_impl(x)
