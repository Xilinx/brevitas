"""
Copyright (C) 2024,     Advanced Micro Devices, Inc.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

# mypy: allow-untyped-defs
from typing import List, Optional

import torch
from torch import Tensor

try:
    from torch.optim.optimizer import _default_to_fused_or_foreach
    from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
except:
    _default_to_fused_or_foreach = None
    _get_fused_kernels_supported_devices = None
try:
    from torch.optim.optimizer import _use_grad_for_differentiable
except:
    # Ensure backward compatibility with PyTorch < 1.13.0
    _use_grad_for_differentiable = torch.no_grad

from torch.optim.optimizer import Optimizer

__all__ = ["SignSGD", "sign_sgd"]


class SignSGD(Optimizer):
    r"""Implements signed stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                               \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                        \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma \text{sign}(g_t)      \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \text{sign}(g_t)      \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SignSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True

            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(p.device.type in fused_supported_devices and torch.is_floating_point(p)
                       for pg in self.param_groups
                       for p in pg["params"]):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}.")
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(group, params, grads, momentum_buffer_list)

            sign_sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def sign_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs Sign SGD algorithm computation.

    See :class:`~torch.optim.SignSGD` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting() and _default_to_fused_or_foreach is not None:
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sign_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sign_sgd
    else:
        func = _single_tensor_sign_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_sign_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        param.add_(torch.sign(grad), alpha=-lr)


def _multi_tensor_sign_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],
        with_indices=True  # type: ignore[list-item]
    )
    for (
        device_params,
        device_grads,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch._utils.is_compiling():
                grads_x_lr = torch._foreach_mul(torch._foreach_sign(device_grads), -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, torch._foreach_sign(device_grads), alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(torch.sign(device_grads[i]), alpha=-lr)


def _fused_sign_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    raise NotImplementedError("Fused SignSGD is not implemented.")
