# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.gptq import GPTQ
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed


def _project_onto_l1_ball(x, eps=1.0):
    """
    Vectorized L1 ball projection.

    Adapted from https://github.com/AozhongZhang/MagR, released under the following LICENSE:

    MIT License

    Copyright (c) 2025 Aozhong Zhang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x


def _power_iteration(H, steps: int, eps: float = 1e-12):
    """
    Power iteration to compute the maximum singular value for the Hessian.
    """
    b_k = torch.rand(H.shape[1], device=H.device)
    for _ in range(steps):
        b_k1 = torch.mv(H, b_k)  # Calculate the matrix-by-vector product H*b_k
        b_k1_norm = torch.norm(b_k1)  # Calculate the norm
        b_k = b_k1 / (b_k1_norm + eps)  # Re normalize the vector
    max_singular_value = torch.dot(b_k, torch.mv(H, b_k))
    return max_singular_value


class MagR(GPTQ):
    """
    Implementation of MagR algorithm for PTQ pre-processing.
    """

    def __init__(
            self,
            layer,
            name,
            len_parallel_layers,
            create_weight_orig,
            gradient_steps: int = 200,
            power_steps: int = 30,
            alpha: float = 0.01) -> None:
        # Note: using GPxQ initialization to avoid blocksize initialization and the
        # torch versioning assertion
        GPxQ.__init__(self, layer, name, None, len_parallel_layers, create_weight_orig)
        self.gradient_steps = gradient_steps
        self.power_steps = power_steps
        self.alpha = alpha

        # Initialize covariance matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device='cpu',
                             dtype=torch.float32,
                             pin_memory=torch.cuda.is_available())
        self.B = torch.zeros((self.groups, self.columns, self.columns),
                             device='cpu',
                             dtype=torch.float32,
                             pin_memory=torch.cuda.is_available())
        self.nsamples = 0

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input
        # Workaround to avoid duplication with GPTQ and MagR, will have the same method
        # across GPxQ classes
        self.compute_iterative_covariance(module, input, current_layer)

    def single_layer_update(self):
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        weight = self.layer.weight.data
        if self.create_weight_orig:
            weight_orig = self.layer.weight_orig.data
        else:
            weight_orig = weight.detach().clone()

        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)  # This performs a view
                weight_orig = weight_orig.transpose(1, 0)
            weight = weight.flatten(1)
            weight_orig = weight_orig.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]
        weight_orig = weight_orig.view(
            self.groups, -1, weight_orig.shape[-1])  # [Groups, OC/Groups, IC]
        self.H = self.H.to(dev)
        for group_index in range(self.groups):
            # approximate maximum singular value (ie, matrix L2 norm)
            eta = 1. / _power_iteration(self.H[group_index], steps=self.power_steps)
            alpha = self.alpha / (eta * torch.linalg.norm(self.H[group_index], ord=1))
            wk = weight[group_index].to(torch.float32)
            gk = weight_orig[group_index].to(torch.float32)  # ground
            for _ in tqdm(range(self.gradient_steps), leave=False):
                vk = wk - eta * (wk - gk).matmul(
                    self.H[group_index])  # argument of the proximal operator
                wk = vk - alpha * _project_onto_l1_ball(vk / alpha)  # update via proximal operator
                weight[group_index] = wk.to(dtype)  # downcast
                assert torch.isfinite(weight[group_index]).all()
        del self.H  # free memory
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class magr_mode(gpxq_mode):
    """
    Apply MagR algorithm, https://arxiv.org/abs/2406.00800

    Args:
        model (Module): The model to pre-process with MagR
        alpha (float): The L-infty norm penalty for MagR. Default: 0.1
        num_steps (int): The number of gradient steps for MagR algorithm. Default: 10
        group_of_parallel_layers (Optional, List[str]): List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply MagR inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            MagR. These weights will be used anytime quantization is disabled. Default: True
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with magr_mode(model) as magr:
        >>>         magr_model = magr.model
        >>>         for i in tqdm(range(magr.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 magr_model(img)
        >>>             magr.update()
    """

    def __init__(
            self,
            model,
            alpha: float = 0.1,
            num_steps: int = 10,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            return_forward_output: bool = False) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(
            model=model,
            group_of_parallel_layers=group_of_parallel_layers,
            inplace=inplace,
            create_weight_orig=create_weight_orig,
            return_forward_output=return_forward_output)
        self.num_steps = num_steps
        self.alpha = alpha

    def _is_module_supported(self, module):
        return isinstance(module, (nn.Linear, *SUPPORTED_CONV_OP))

    def update(self):
        for name in tqdm(self.current_layer.layer_names, desc='Updating weights...', leave=True):
            self.gpxq_layers[name].single_layer_update()
            self.hook_dict[name].remove()
        self.current_layer.layer_names.clear()

    def catch_stopfwd(self, *args, **kwargs):
        self.orig_forward(*args, **kwargs)
        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            out = self.orig_forward(*args, **kwargs)
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = False
            return out

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        return MagR(
            layer=layer,
            name=name,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            gradient_steps=self.num_steps,
            alpha=self.alpha)
