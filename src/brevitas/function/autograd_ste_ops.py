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
Implementation of various torch.autograd.Function with straight-through estimators.
"""

from typing import Tuple

import torch
from torch.autograd import Function
from torch import Tensor

from brevitas.function.ops import tensor_clamp, binary_sign, round_to_zero

__all__ = [
    'round_ste_impl',
    'binary_sign_ste_impl',
    'ternary_sign_ste_impl',
    'floor_ste_impl',
    'ceil_ste_impl',
    'round_to_zero_ste_impl',
    'scalar_clamp_min_ste_impl',
    'scalar_clamp_ste_impl',
    'tensor_clamp_ste_impl',
    'abs_binary_sign_grad_impl'
]


class ScalarClampSteFn(Function):
    """
    Autograd function that implements torch.clamp with a straight-through gradient estimator for
    the gradient of y w.r.t. to x, while the gradient of y w.r.t. to min_val and max_val is always
    None.

    Notes:
        ScalarClampSteFn.apply is exported as scalar_clamp_ste_impl.

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = scalar_clamp_ste_impl(x, -1.0, 1.0)
        >>> y
        tensor([ 1.0000,  0.4000, -1.0000], grad_fn=<ScalarClampSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True

    """
    
    @staticmethod
    def forward(ctx, x: Tensor, min_val: float, max_val: float) -> Tensor:
        y = torch.clamp(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        return grad_y, None, None


class ScalarClampMinSteFn(Function):
    """
    Autograd function that implements torch.clamp_min with a straight-through gradient estimator for
    the gradient of y w.r.t. to x, while the gradient of y w.r.t. to min_val is always None.

    Notes:
        TensorClampSteFn.apply is exported as scalar_clamp_min_ste_impl.

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = scalar_clamp_min_ste_impl(x, -1.0)
        >>> y
        tensor([ 1.5000,  0.4000, -1.0000], grad_fn=<ScalarClampMinSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """
    
    @staticmethod
    def forward(ctx, x: Tensor, min_val: float) -> Tensor:
        y = torch.clamp_min(x, min_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None]:
        return grad_y, None


class TensorClampSteFn(Function):
    """ 
    Autograd function that implements tensor_clamp with a straight-through gradient estimator for
    the gradient of y w.r.t. to x, while the gradient of y w.r.t. to min_val and max_val is always
    None.
    See :func:`~brevitas.function.ops.tensor_clamp` for further details.

    Notes:
        TensorClampSteFn.apply is exported as tensor_clamp_ste_impl.

    Examples:
        >>> x = torch.tensor([1.5, 0.4, -1.5], requires_grad=True)
        >>> y = tensor_clamp_ste_impl(x, torch.tensor([-1.0, -0.5, -0.5]), torch.tensor([1.0, 0.5, 0.5]))
        >>> y
        tensor([ 1.0000,  0.4000, -0.5000], grad_fn=<TensorClampSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1, 0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
        y = tensor_clamp(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        return grad_y, None, None


class RoundToZeroSteFn(Function):
    """
    Autograd function that implements rounding towards zero with a straight-through gradient estimator.
    See :func:`~brevitas.function.ops.round_to_zero` for further details.

    Notes:
        RoundToZeroSteFn.apply is exported as round_to_zero_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = round_to_zero_ste_impl(x)
        >>> y
        tensor([ 1., -1.], grad_fn=<RoundToZeroSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = round_to_zero(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class CeilSteFn(Function):
    """ 
    Autograd function that implements torch.ceil with a straight-through gradient estimator.

    Notes:
        CeilSteFn.apply is exported as ceil_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = ceil_ste_impl(x)
        >>> y
        tensor([ 2., -1.], grad_fn=<CeilSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.ceil(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class FloorSteFn(Function):
    """ 
    Autograd function that implements torch.floor with a straight-through gradient estimator.

    Notes:
        FloorSteFn.apply is exported as floor_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = floor_ste_impl(x)
        >>> y
        tensor([ 1., -2.], grad_fn=<FloorSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.floor(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class BinarySignSteFn(Function):
    """ 
    Autograd function that implements binary_sign with a straight-through gradient estimator.
    See :func:`~brevitas.function.ops.binary_sign` for further details.

    Notes:
        BinarySignSteFn.apply is exported as binary_sign_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, 0.0, -0.5], requires_grad=True)
        >>> y = binary_sign_ste_impl(x)
        >>> y
        tensor([ 1.,  1., -1.], grad_fn=<BinarySignSteFnBackward>)
        >>> grad = torch.tensor([0.1, 0.2, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = binary_sign(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class TernarySignSteFn(Function):
    """ 
    Autograd function that implements torch.sign with a straight-through gradient estimator.

    Notes:
        TernarySignSteFn.apply is exported as ternary_sign_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, 0.0, -0.5], requires_grad=True)
        >>> y = ternary_sign_ste_impl(x)
        >>> y
        tensor([ 1.,  0., -1.], grad_fn=<TernarySignSteFnBackward>)
        >>> grad = torch.tensor([0.1, 0.2, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.sign(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class RoundSteFn(Function):
    """
    Autograd function that implements torch.round with a straight-through gradient estimator.

    Notes:
        RoundSteFn.apply is exported as round_ste_impl.

    Examples:
        >>> x = torch.tensor([1.7, -1.7], requires_grad=True)
        >>> y = round_ste_impl(x)
        >>> y
        tensor([ 2., -2.], grad_fn=<RoundSteFnBackward>)
        >>> grad = torch.tensor([0.1, -0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y


class AbsBinarySignGradFn(Function):
    """
        Autograd function that implements torch.abs with a binary-sign backward, in order to have
        subgradient 1 in 0. Compare with torch.abs' subgradient of 0 in 0.

    Notes:
        AbsBinarySignGradFn.apply is exported as abs_binary_sign_grad_impl.

    Examples:
        >>> x = torch.tensor([0.0], requires_grad=True)
        >>> y = abs_binary_sign_grad_impl(x)
        >>> y
        tensor([0.], grad_fn=<AbsBinarySignGradFnBackward>)
        >>> grad = torch.tensor([0.1])
        >>> y.backward(grad)
        >>> (x.grad == grad).all().item()
        True
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(binary_sign(x).type(torch.int8))  # save some memory
        y = torch.abs(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        binary_sign, = ctx.saved_tensors
        return binary_sign.float() * grad_y


round_ste_impl = RoundSteFn.apply
binary_sign_ste_impl = BinarySignSteFn.apply
ternary_sign_ste_impl = TernarySignSteFn.apply
floor_ste_impl = FloorSteFn.apply
ceil_ste_impl = CeilSteFn.apply
round_to_zero_ste_impl = RoundToZeroSteFn.apply
scalar_clamp_min_ste_impl = ScalarClampMinSteFn.apply
scalar_clamp_ste_impl = ScalarClampSteFn.apply
tensor_clamp_ste_impl = TensorClampSteFn.apply
abs_binary_sign_grad_impl = AbsBinarySignGradFn.apply