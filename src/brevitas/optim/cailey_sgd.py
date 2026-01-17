# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This file was adapted from a file in a repository
# licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# Original author: (c) Meta Platforms, Inc. and affiliates.
# Source: https://github.com/facebookresearch/SpinQuant/blob/main/train_utils/optimizer.py

# This code is originally from: https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py

import random

import torch
from torch.optim.optimizer import Optimizer


def unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = 1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def cayley_loop(X, W, tan_vec, t, iters=5):
    Y = X + t * tan_vec
    for _ in range(iters):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()


def qr_retraction(tan_vec):
    tan_vec.t_()
    dtype = tan_vec.dtype
    # torch.linalg.qr is not implemented for 'Half'
    q, r = torch.linalg.qr(tan_vec.to(torch.float32))
    q, r = q.to(dtype=dtype), r.to(dtype=dtype)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q


epsilon = 1e-8


class CaileySGD(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        momentum: int = 0,
        dampening: int = 0,
        weight_decay: int = 0,
        nesterov: bool = False,
        stiefel: bool = False,
        iters: int = 5,
        grad_clip: bool = None,
        dtype: str = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=0,
            iters=iters,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CaileySGD, self).__init__(params, defaults)
        self.dtype = getattr(torch, dtype) if dtype is not None else dtype

    def __setstate__(self, state) -> None:
        super(CaileySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]
            iters = group["iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                param_state = self.state[p]
                # Store a copy of weights in desired dtype if it is different from param dtype
                if self.dtype is not None and self.dtype != param.dtype:
                    if "weight_buffer" not in param_state:
                        param_state["weight_buffer"] = param.clone().to(self.dtype)
                    param = param_state["weight_buffer"]

                unity = param.view(p.size()[0], -1)
                unity, _ = unit(unity)
                if stiefel and unity.size()[0] <= unity.size()[1]:

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)
                    if self.dtype is not None:
                        g = g.to(self.dtype)

                    lr = group["lr"]

                    # if momentum is used, initialize the momentum buffer
                    V = 0.
                    if momentum != 0:
                        if "momentum_buffer" not in param_state:
                            param_state["momentum_buffer"] = torch.zeros_like(g.t())
                        V = param_state["momentum_buffer"]

                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 1. / (matrix_norm_one(W) + epsilon)
                    alpha = min(t, lr)

                    p_new = cayley_loop(unity.t(), W, V, alpha, iters)
                    param.copy_(p_new)  # update shadow weights
                    p.data.copy_(p_new.view(p.size()).to(p.data.dtype))  # update param.data
                    # update momentum buffer if momentum is used
                    if momentum != 0:
                        V.copy_(torch.mm(W, unity.t()))  # n-by-p

                else:

                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]
                    d_p = p.grad.data
                    #  defined.
                    try:
                        if weight_decay != 0:
                            #  defined.
                            d_p.add_(weight_decay, p.data)
                    except:
                        pass
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            #  always defined.
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        #  defined.
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group["lr"], d_p)

        return loss
