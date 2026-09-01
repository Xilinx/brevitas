# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.gptq import GPTQ
from brevitas.graph.gpxq import FunctionalGPxQBatch
from brevitas.graph.gpxq import FunctionalLinearTarget
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import power_iteration
from brevitas.utils.torch_utils import StopFwdException


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
            alpha: float = 0.01,
            device='cpu',
            dtype=torch.float32) -> None:
        # Note: using GPxQ initialization to avoid blocksize initialization and the
        # torch versioning assertion
        GPxQ.__init__(
            self, layer, name, None, len_parallel_layers, create_weight_orig, device, dtype)
        self.gradient_steps = gradient_steps
        self.power_steps = power_steps
        self.alpha = alpha

        # Initialize covariance matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device=self.device,
                             dtype=self.dtype)
        if self.use_intermediate_buffer:
            self.B = torch.zeros((self.groups, self.columns, self.columns),
                                 device=self.device,
                                 dtype=self.dtype,
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
        functional_weight_before = (
            weight.detach().clone() if isinstance(self.layer, FunctionalLinearTarget) else None)
        if self.create_weight_orig:
            weight_orig = self.layer.weight_orig.data
        else:
            weight_orig = weight.detach().clone()

        dev = weight.device
        weight_orig = weight_orig.to(dev)

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
        failed = False
        for group_index in range(self.groups):
            # approximate maximum singular value (ie, matrix L2 norm)
            singular_value = power_iteration(self.H[group_index], steps=self.power_steps)
            matrix_norm = torch.linalg.norm(self.H[group_index], ord=1)
            if singular_value <= 0 or matrix_norm <= 0:
                warnings.warn(
                    f'MagR will not be applied to layer {self.name}: empty covariance matrix.')
                failed = True
                continue
            eta = 1. / singular_value
            alpha = self.alpha / (eta * matrix_norm)
            wk = weight[group_index].to(self.dtype)
            gk = weight_orig[group_index].to(self.dtype)  # ground
            for _ in tqdm(range(self.gradient_steps), leave=False):
                vk = wk - eta * (wk - gk).matmul(
                    self.H[group_index])  # argument of the proximal operator
                wk = vk - alpha * _project_onto_l1_ball(vk / alpha)  # update via proximal operator
                weight[group_index] = wk.to(dtype)  # downcast
                if not torch.isfinite(weight[group_index]).all():
                    if functional_weight_before is not None:
                        self.layer.writeback(functional_weight_before)
                    warnings.warn(
                        f'MagR update for layer {self.name} produced non-finite weights; '
                        'restoring its pre-MagR weights.')
                    del self.H
                    if hasattr(self.layer, 'offload_params'):
                        self.layer.offload_params(self.layer)
                    return True
        del self.H  # free memory
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
        return failed

    @staticmethod
    def batched_layer_update(optimizers):
        """Apply MagR to a compatible batch of functional expert matrices."""
        batch = FunctionalGPxQBatch(optimizers)
        first = optimizers[0]
        if any(optimizer.columns != first.columns or
               optimizer.gradient_steps != first.gradient_steps or
               optimizer.power_steps != first.power_steps or optimizer.alpha != first.alpha
               for optimizer in optimizers):
            raise ValueError('Batched functional MagR requires compatible expert optimizers.')
        for optimizer in optimizers:
            if optimizer.use_intermediate_buffer:
                del optimizer.B

        targets = batch.targets
        weight = batch.weight
        weight_orig = torch.stack([
            target.weight_orig.to(weight.device)
            if optimizer.create_weight_orig else target.weight.detach().clone() for target,
            optimizer in zip(targets, optimizers)])
        hessian = batch.pop_buffer('H', weight.device)

        scale = hessian.amax(dim=(-2, -1)).abs()
        valid = scale > 0
        normalized = hessian / torch.where(valid, scale, torch.ones_like(scale))[:, None, None]
        generator = torch.Generator(device=weight.device).manual_seed(42)
        vector = torch.rand(
            first.columns, device=weight.device, dtype=first.dtype,
            generator=generator).expand(len(optimizers), -1).clone()
        eps = torch.finfo(first.dtype).eps
        for _ in range(first.power_steps):
            next_vector = torch.bmm(normalized, vector.unsqueeze(2)).squeeze(2)
            vector = next_vector / (
                torch.linalg.vector_norm(next_vector, dim=1, keepdim=True).clamp_min(eps))
        singular_value = torch.bmm(vector.unsqueeze(1), torch.bmm(
            normalized, vector.unsqueeze(2))).flatten() * scale
        matrix_norm = torch.linalg.matrix_norm(hessian, ord=1)
        valid &= (singular_value > 0) & (
            matrix_norm > 0) & torch.isfinite(singular_value) & torch.isfinite(matrix_norm)
        failed = [targets[index] for index in torch.where(~valid)[0].tolist()]
        if not valid.any():
            return failed

        indices = torch.where(valid)[0]
        targets = [targets[index] for index in indices.tolist()]
        weight = weight.index_select(0, indices)
        weight_orig = weight_orig.index_select(0, indices)
        hessian = hessian.index_select(0, indices)
        eta = 1. / singular_value.index_select(0, indices)
        alpha = first.alpha / (eta * matrix_norm.index_select(0, indices))
        wk = weight.to(first.dtype)
        gk = weight_orig.to(first.dtype)
        rows = wk.shape[1]
        for _ in range(first.gradient_steps):
            vk = wk - eta[:, None, None] * torch.bmm(wk - gk, hessian)
            projected = _project_onto_l1_ball((vk / alpha[:, None, None]).reshape(
                -1, first.columns)).reshape(len(targets), rows, first.columns)
            wk = vk - alpha[:, None, None] * projected
        failed.extend(FunctionalGPxQBatch.writeback(targets, wk.to(weight.dtype)))
        return failed


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
        device (str): Device the buffers are stored on. Default: cpu
        dtype (torch.dtype): Datatype the buffers are stored in. Default: torch.float32

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
        return_forward_output: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        functional_state=None,
        min_samples: int = 0,
        insufficient_samples: str = 'rtn',
        expert_batch_size: int = 1,
        functional_linear_functions: Sequence[Callable] = (),
        functional_matmul_functions: Sequence[Callable] = (),
        functional_grouped_mm_functions: Sequence[Callable] = ()
    ) -> None:
        super().__init__(
            model=model,
            group_of_parallel_layers=group_of_parallel_layers,
            inplace=inplace,
            create_weight_orig=create_weight_orig,
            return_forward_output=return_forward_output,
            device=device,
            dtype=dtype,
            functional_state=functional_state,
            min_samples=min_samples,
            insufficient_samples=insufficient_samples,
            expert_batch_size=expert_batch_size,
            functional_linear_functions=functional_linear_functions,
            functional_matmul_functions=functional_matmul_functions,
            functional_grouped_mm_functions=functional_grouped_mm_functions)
        self.num_steps = num_steps
        self.alpha = alpha

    def _update_functional_targets(self, targets, progress) -> int:
        if self.expert_batch_size == 1:
            return super()._update_functional_targets(targets, progress)
        failed = []
        for start in range(0, len(targets), self.expert_batch_size):
            batch_targets = targets[start:start + self.expert_batch_size]
            optimizers = [self.gpxq_layers[target.name] for target in batch_targets]
            failed.extend(MagR.batched_layer_update(optimizers))
            progress.set_postfix(batch=len(batch_targets), failed=len(failed))
            progress.update(len(batch_targets))
        return len(failed)

    def _is_module_supported(self, module):
        return isinstance(module, (nn.Linear, *SUPPORTED_CONV_OP))

    def update(self):
        super().update()

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            if self.functional_source is not None:
                self.functional_source.restart_call_sequence()
            if self.functional_session is not None:
                self.functional_session.enabled = False
            try:
                out = self.orig_forward(*args, **kwargs)
            finally:
                if self.functional_session is not None:
                    self.functional_session.enabled = True
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
            alpha=self.alpha,
            device=self.device,
            dtype=self.dtype)
