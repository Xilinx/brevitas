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

from brevitas.function.ops import tensor_clamp, binary_sign, round_to_zero, tensor_clamp_, dpu_round

__all__ = [
    'ScalarClampSteFn',
    'ScalarClampMinSteFn',
    'TensorClampSteFn',
    'InplaceTensorClampSteFn',
    'RoundToZeroSteFn',
    'CeilSteFn',
    'FloorSteFn',
    'BinarySignSteFn',
    'TernarySignSteFn',
    'RoundSteFn',
    'AbsBinarySignGradFn',
    'DPURoundSteFn',
    'round_ste_impl',
    'binary_sign_ste_impl',
    'ternary_sign_ste_impl',
    'floor_ste_impl',
    'ceil_ste_impl',
    'round_to_zero_ste_impl',
    'scalar_clamp_min_ste_impl',
    'scalar_clamp_ste_impl',
    'tensor_clamp_ste_impl',
    'abs_binary_sign_grad_impl',
    'dpu_round_ste_impl'
]


class ScalarClampSteFn(Function):
    """
    ``ScalarClampSteFn.apply(*args)`` is first aliased to :func:`scalar_clamp_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.scalar_clamp_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.scalar_clamp_ste` and invoked when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.scalar_clamp_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_val: float, max_val: float) -> Tensor:
        y = torch.clamp(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        return grad_y, None, None

    @staticmethod
    def symbolic(g, x: Tensor, min_val: float, max_val: float):
        y = g.op('Clip', x, torch.tensor(min_val), torch.tensor(max_val))
        return y


class ScalarClampMinSteFn(Function):
    """
    Autograd function that implements ``torch.clamp_min`` with a straight-through gradient estimator
    for the gradient of y w.r.t. to x, while the gradient of y w.r.t. to ``min_val`` is always
    ``None``.

    ``ScalarClampMinSteFn.apply(*args)`` is first aliased to :func:`scalar_clamp_min_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.scalar_clamp_min_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.scalar_clamp_min_ste` and invoked when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.scalar_clamp_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_val: float) -> Tensor:
        y = torch.clamp_min(x, min_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None]:
        return grad_y, None

    @staticmethod
    def symbolic(g, x: Tensor, min_val: float):
        y = g.op('Clip', x, torch.tensor(min_val))
        return y


class TensorClampSteFn(Function):
    """ 
    Autograd function that implements :func:`~brevitas.function.ops.tensor_clamp` with a
    straight-through gradient estimator for the gradient of y w.r.t. to x, while the gradient of y
    w.r.t. to min_val and max_val is always None.

    ``TensorClampSteFn.apply(*args)`` is first aliased to :func:`tensor_clamp_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.tensor_clamp_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.tensor_clamp` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.tensor_clamp` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
        y = tensor_clamp(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        return grad_y, None, None

    @staticmethod
    def symbolic(g, x: Tensor, min_val: Tensor, max_val: Tensor):
        upper_cond = g.op('Greater', x, max_val)
        y = g.op('Where', upper_cond, max_val, x)
        lower_cond = g.op('Less', y, min_val)
        y = g.op('Where', lower_cond, min_val, y)
        return y


class InplaceTensorClampSteFn(Function):
    """
    Autograd function that implements :func:`~brevitas.function.ops.tensor_clamp_` with a
    straight-through gradient estimator for the gradient of y w.r.t. to x, while the gradient of y
    w.r.t. to min_val and max_val is always None.

    ``InplaceTensorClampSteFn.apply(*args)`` is first aliased to
    :func:`tensor_clamp_ste_impl_(*args)
    <brevitas.function.autograd_ste_ops.tensor_clamp_ste_impl_>` and then wrapped by
    :func:`~brevitas.function.ops_ste.tensor_clamp_` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.tensor_clamp_` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor, min_val: Tensor, max_val: Tensor) -> Tensor:
        y = tensor_clamp_(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        return grad_y, None, None

    @staticmethod
    def symbolic(g, x: Tensor, min_val: Tensor, max_val: Tensor):
        upper_cond = g.op('Greater', x, max_val)
        y = g.op('Where', upper_cond, max_val, x)
        lower_cond = g.op('Less', y, min_val)
        y = g.op('Where', lower_cond, min_val, y)
        return y


class RoundToZeroSteFn(Function):
    """
    Autograd function that implements :func:`~brevitas.function.ops.round_to_zero` with a
    straight-through gradient estimator.

    ``RoundToZeroSteFn.apply(*args)`` is first aliased to :func:`round_to_zero_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.round_to_zero_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.round_to_zero_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.round_to_zero_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = round_to_zero(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        abs = g.op('Abs', x)
        sign = g.op('Sign', x)
        floor = g.op('Floor', abs)
        y = g.op('Mul', sign, floor)
        return y


class DPURoundSteFn(Function):
    """
    Autograd function that implements :func:`~brevitas.function.ops.dpu_round` with a
    straight-through gradient estimator.

    ``DPURoundSteFn.apply(*args)`` is first aliased to :func:`dpu_round_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.dpu_round_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.dpu_round_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.dpu_round_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = dpu_round(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        raise NotImplementedError


class CeilSteFn(Function):
    """ 
    Autograd function that implements :func:`torch.ceil` with a straight-through gradient estimator.

    ``CeilSteFn.apply(*args)`` is first aliased to :func:`ceil_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.ceil_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.ceil_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.ceil_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.ceil(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        y = g.op('Ceil', x)
        return y


class FloorSteFn(Function):
    """ 
    Autograd function that implements :func:`torch.floor` with a straight-through gradient estimator.

    ``FloorSteFn.apply(*args)`` is first aliased to :func:`floor_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.floor_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.floor_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.floor_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.floor(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        y = g.op('Floor', x)
        return y


class BinarySignSteFn(Function):
    """ 
    Autograd function that implements :func:`~brevitas.function.ops.binary_sign` with a
    straight-through gradient estimator.

   ``BinarySignSteFn.apply(*args)`` is first aliased to :func:`binary_sign_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.binary_sign_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.binary_sign_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.binary_sign_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = binary_sign(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        # requires ONNX opset >= 12
        positive_mask = g.op('GreaterOrEqual', x, torch.tensor(0.))
        negative_mask = g.op('Less', x, torch.tensor(0.))
        positive_mask = g.op('Cast', positive_mask, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        negative_mask = g.op('Cast', negative_mask, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        y = g.op('Sub', positive_mask, negative_mask)
        return y


class TernarySignSteFn(Function):
    """ 
    Autograd function that implements :func:`torch.sign` with a straight-through gradient estimator.

    ``TernarySignSteFn.apply(*args)`` is first aliased to :func:`ternary_sign_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.ternary_sign_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.ternary_sign_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.ternary_sign_ste` for details on the interface and
    examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.sign(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        y = g.op('Sign', x)
        return y


class RoundSteFn(Function):
    """
    Autograd function that implements :func:`torch.round` with a straight-through gradient
    estimator.

    ``RoundSteFn.apply(*args)`` is first aliased to :func:`round_ste_impl(*args)
    <brevitas.function.autograd_ste_ops.round_ste_impl>` and then wrapped by
    :func:`~brevitas.function.ops_ste.round_ste` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.round_ste` for details on the interface and examples.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> Tensor:
        return grad_y

    @staticmethod
    def symbolic(g, x: Tensor):
        # Requires ONNX opset >= 11
        y = g.op('Round', x)
        return y


class AbsBinarySignGradFn(Function):
    """
    Autograd function that implements :func:`torch.abs` with a binary-sign backward, in order to
    have subgradient 1 in 0. Compare with :func:`torch.abs`' subgradient of 0 in 0.

    ``AbsBinarySignGradFn.apply(*args)`` is first aliased to :func:`abs_binary_sign_grad(*args)
    <brevitas.function.autograd_ste_ops.abs_binary_sign_grad>` and then wrapped by
    :func:`~brevitas.function.abs_binary_sign_grad` when env ``BREVITAS_JIT=0``.
    See :func:`~brevitas.function.ops_ste.abs_binary_sign_grad` for details on the interface and
    examples.
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

    @staticmethod
    def symbolic(g, x: Tensor):
        y = g.op('Abs', x)
        return y


#: Alias for :class:`RoundSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.RoundSteFn>`
round_ste_impl = RoundSteFn.apply

#: Alias for :class:`BinarySignSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.BinarySignSteFn>`
binary_sign_ste_impl = BinarySignSteFn.apply

#: Alias for :class:`TernarySignSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.TernarySignSteFn>`
ternary_sign_ste_impl = TernarySignSteFn.apply

#: Alias for :class:`FloorSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.FloorSteFn>`
floor_ste_impl = FloorSteFn.apply

#: Alias for :class:`CeilSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.CeilSteFn>`
ceil_ste_impl = CeilSteFn.apply

#: Alias for :class:`RoundToZeroSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.RoundToZeroSteFn>`
round_to_zero_ste_impl = RoundToZeroSteFn.apply

#: Alias for :class:`DPURoundSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.DPURoundSteFn>`
dpu_round_ste_impl = DPURoundSteFn.apply

#: Alias for :class:`ScalarClampMinSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.ScalarClampMinSteFn>`
scalar_clamp_min_ste_impl = ScalarClampMinSteFn.apply

#: Alias for :class:`ScalarClampSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.ScalarClampSteFn>`
scalar_clamp_ste_impl = ScalarClampSteFn.apply

#: Alias for :class:`TensorClampSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.TensorClampSteFn>`
tensor_clamp_ste_impl = TensorClampSteFn.apply

#: Alias for :class:`InplaceTensorClampSteFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.InplaceTensorClampSteFn>`
tensor_clamp_ste_impl_ = InplaceTensorClampSteFn.apply

#: Alias for :class:`AbsBinarySignGradFn.apply(*args)
#: <brevitas.function.autograd_ste_ops.AbsBinarySignGradFn>`
abs_binary_sign_grad_impl = AbsBinarySignGradFn.apply
