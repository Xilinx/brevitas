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

import torch


class scalar_clamp_ste_fn(torch.autograd.Function):
    """ Autograd function that implements scalar_clamp with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.scalar_clamp_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float, max_val: float):
        """
        """
        y = torch.clamp(x, min_val, max_val)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y, None, None


class scalar_clamp_min_ste_fn(torch.autograd.Function):
    """ Autograd function that implements scalar_clamp_min with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.scalar_clamp_min_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float):
        """
        """
        y = torch.clamp_min(x, min_val)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y, None


class round_to_zero_ste_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.sign(x) * torch.floor(torch.abs(x))
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y, None, None


class tensor_clamp_ste_fn(torch.autograd.Function):
    """ Autograd function that implements tensor_clamp with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.tensor_clamp_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        """
        """
        y = torch.where(x > max_val, max_val, x)
        y = torch.where(y < min_val, min_val, y)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y, None, None


class ceil_ste_fn(torch.autograd.Function):
    """ Autograd function that implements ceil_ste with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.ceil_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.ceil(x)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y


class floor_ste_fn(torch.autograd.Function):
    """ Autograd function that implements floor_ste with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.floor_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.floor(x)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y


class binary_sign_ste_fn(torch.autograd.Function):
    """ Autograd function that implements binary_sign_ste with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.binary_sign_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        positive_mask = torch.ge(x, 0.0)
        negative_mask = torch.lt(x, 0.0)
        y = positive_mask.float() - negative_mask.float()
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y


class ternary_sign_ste_fn(torch.autograd.Function):
    """ Autograd function that implements ternary_sign_ste with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.ternary_sign_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.sign(x)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y


class round_ste_fn(torch.autograd.Function):
    """ Autograd function that implements round_ste with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.round_ste` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.round(x)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y


class round_to_zero_fn(torch.autograd.Function):
    """ Autograd function that implements round_to_zero with a straight through estimator

    Look at the documentation of :func:`~brevitas.function.ops_ste.round_to_zero` for further details.

    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        y = torch.round(x)
        torch.sign(x) * torch.floor(torch.abs(x))
        return y
    @staticmethod
    def backward(ctx, grad_y):
        """
        """
        return grad_y