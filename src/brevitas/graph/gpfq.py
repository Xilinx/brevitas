# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import math
from typing import List
from typing import Optional
import warnings

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpxq import FunctionalGPxQBatch
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.utils.torch_utils import StopFwdException


class GPFQ(GPxQ):
    """
    Optimized greedy path following quantization (GPFQ)

    See `Post-training Quantization for Neural Networks with Provable Guarantees`

    https://epubs.siam.org/doi/abs/10.1137/22M1511709
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            device='cpu',
            dtype=torch.float32) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, device, dtype)
        # Initialize covariance matrices. We need them in float32
        # H = \hat{X} \hat{X}^T
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device=self.device,
                             dtype=self.dtype)
        # G = \hat{X} X^T
        self.G = torch.zeros((self.groups, self.columns, self.columns),
                             device=self.device,
                             dtype=self.dtype)
        if self.use_intermediate_buffer:
            self.B = torch.zeros((self.groups, self.columns, self.columns),
                                 device=self.device,
                                 dtype=self.dtype,
                                 pin_memory=torch.cuda.is_available())
        self.nsamples = 0

        self.quant_input = None

        self.create_weight_orig = create_weight_orig  # not saved by base class

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        is_quant_enabled = not module.reference_pass if hasattr(
            module, 'reference_pass') else module.weight_quant.is_quant_enabled

        # NOTE: batch_size = seqlen for language models here
        inp_processed = self.process_input(input)  # [groups, in_features, batch_size]
        batch_size = inp_processed.shape[-1]

        # Normalizing for numerical stability
        inp_processed = math.sqrt(1 / batch_size) * inp_processed.to(self.dtype)

        # NOTE: in the gpfq_mode context manager, we first collect quant inputs, then
        # we collect float inputs for the same batch. We assume this pattern here, but
        # will add a check just in case.

        # if quant is not enabled, then it is the float input; if it is a float input
        # then a quant input has already happened and we can update G
        if not is_quant_enabled:
            # Compute the normalized G matrix
            if self.use_intermediate_buffer:
                self.B.copy_(self.quant_input.bmm(inp_processed.transpose(2, 1)))
                self.G += self.B
            else:
                self.G += self.quant_input.bmm(inp_processed.transpose(2, 1))
            self.quant_input = None  # NOTE: set back to None now that we've used it
        else:
            # Compute the normalized H matrix
            self.nsamples += batch_size
            if self.use_intermediate_buffer:
                self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
                self.H += self.B
            else:
                self.H += inp_processed.bmm(inp_processed.transpose(2, 1))
            # store the quantized input for computing the H matrix
            assert self.quant_input is None
            self.quant_input = inp_processed

        # If we are executing GPFQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            if current_layer.stop_forward:
                raise StopFwdException

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require metadata from input quantizers."
        assert hasattr(self.layer, 'weight_orig'), \
            "Error: GPFQ requires the original weights to be stored, see `create_weight_orig`."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        if self.use_intermediate_buffer:
            del self.B  # free memory

        weight = self.layer.weight.data
        weight_orig = self.layer.weight_orig.data
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

        # Get the diagonals of the covariance matrices here
        permutation_list = []
        for group_index in range(self.groups):
            # If a diagonal element on either covariance matrix is zero, we can set to 0
            # the corresponding column in the weight matrix.
            dead = self.H[group_index].diag() == 0
            weight[group_index, :, dead] = 0
            # Re-order so that weights associated to higher magnitude activations
            # are quantized first if self.act_order is True
            if self.act_order:
                # order w.r.t. the quantized inputs
                perm = torch.argsort(torch.diag(self.H[group_index]), descending=True)
                # Re-order covariance matrices so that weights associated to
                # higher magnitude activations are quantized first
                self.G[group_index] = self.G[group_index, perm, :][:, perm]
                self.H[group_index] = self.H[group_index, perm, :][:, perm]
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(self.H.shape[-1]), device=dev)
            perm = perm.to(weight.device)
            permutation_list.append(perm)

        Dg = torch.zeros((self.groups, self.columns), dtype=self.dtype, device=self.device)
        Dh = torch.zeros((self.groups, self.columns), dtype=self.dtype, device=self.device)
        for group_index in range(self.groups):
            Dg[group_index].copy_(self.G[group_index].diag())
            Dh[group_index].copy_(self.H[group_index].diag())
        # if either norms are 0, the weight is effectively pruned
        Ds = torch.where(Dg * Dh != 0, Dg / Dh, torch.zeros_like(Dg))  # \hat{D}_tt / D_tt

        Lg = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=self.dtype)
        Lh = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=self.dtype)
        for group_index in range(self.groups):
            L0g = torch.tril(self.G[group_index], -1)  # L0
            L0h = torch.tril(self.H[group_index], -1)  # \hat{L0}
            Dhi = torch.where(
                Dh[group_index] != 0, 1. / Dh[group_index],
                torch.zeros_like(Dh[group_index]))  # D^{-1}
            Lg[group_index].copy_(torch.diag(Dhi) @ L0g)
            Lh[group_index].copy_(torch.diag(Dhi) @ L0h)

        del self.H, self.G  # memory management

        for t in range(weight.shape[-1]):
            q_groups = self.get_quant_weights(t, 0, permutation_list, with_quant_history=True)
            for group_index in range(self.groups):
                # t := time step (Lg, Lh, and Ds are re-ordered in time)
                # i := input channel index (weight and error are not re-ordered)
                i = permutation_list[group_index][t]
                w = weight_orig[group_index, :, permutation_list[group_index][:t]].to(self.dtype)
                q = q_groups[group_index].to(self.dtype)
                Lw = w.matmul(Lg[group_index, t, :t])
                Lq = q.matmul(Lh[group_index, t, :t])
                q_arg = Ds[group_index, t] * weight[group_index, :, i].to(self.dtype) + Lw - Lq
                assert not torch.isnan(q_arg).any()
                weight[group_index, :, i] = q_arg.to(dtype)

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)

    @staticmethod
    def batched_layer_update(optimizers):
        """Apply GPFQ to a compatible batch of functional expert matrices."""
        batch = FunctionalGPxQBatch(optimizers)
        first = optimizers[0]
        columns = first.columns
        if any(optimizer.columns != columns or optimizer.act_order != first.act_order
               for optimizer in optimizers):
            raise ValueError('Batched functional GPFQ requires compatible expert optimizers.')
        for optimizer in optimizers:
            if optimizer.layer.weight_quant.requires_quant_input:
                raise RuntimeError(
                    'GPFQ does not support weight quantizers that require input metadata.')
            if optimizer.quant_input is not None:
                raise RuntimeError('GPFQ quantized and reference inputs are unbalanced.')
            if optimizer.use_intermediate_buffer:
                del optimizer.B

        targets = batch.targets
        weight = batch.weight
        weight_orig = torch.stack([target.weight_orig.to(weight.device) for target in targets])
        device = weight.device
        dtype = weight.dtype
        hessian = batch.pop_buffer('H')
        cross_covariance = batch.pop_buffer('G')

        diagonal_h = hessian.diagonal(dim1=-2, dim2=-1)
        dead = diagonal_h == 0
        weight.masked_fill_(dead.to(device).unsqueeze(1), 0)
        if first.act_order:
            permutation = torch.argsort(diagonal_h, dim=-1, descending=True)
            hessian = torch.gather(hessian, 1, permutation.unsqueeze(-1).expand(-1, -1, columns))
            hessian = torch.gather(hessian, 2, permutation.unsqueeze(1).expand(-1, columns, -1))
            cross_covariance = torch.gather(
                cross_covariance, 1, permutation.unsqueeze(-1).expand(-1, -1, columns))
            cross_covariance = torch.gather(
                cross_covariance, 2, permutation.unsqueeze(1).expand(-1, columns, -1))
        else:
            permutation = torch.arange(columns, device=hessian.device).expand(len(optimizers), -1)

        diagonal_g = cross_covariance.diagonal(dim1=-2, dim2=-1)
        diagonal_h = hessian.diagonal(dim1=-2, dim2=-1)
        ds = torch.where(
            diagonal_g * diagonal_h != 0, diagonal_g / diagonal_h, torch.zeros_like(diagonal_g))
        reciprocal_h = torch.where(diagonal_h != 0, 1. / diagonal_h, torch.zeros_like(diagonal_h))
        lg = reciprocal_h.unsqueeze(2) * torch.tril(cross_covariance, diagonal=-1)
        lh = reciprocal_h.unsqueeze(2) * torch.tril(hessian, diagonal=-1)
        del hessian, cross_covariance

        permutation_weight = permutation.to(device)

        def quantize(value):
            return batch.quantize(targets, value)

        def gather_columns(value, indices):
            return torch.gather(value, 2, indices.unsqueeze(1).expand(-1, value.shape[1], -1))

        def scatter_column(value, indices, update):
            return value.scatter(
                2, indices[:, None, None].expand(-1, value.shape[1], 1), update.unsqueeze(2))

        for step in range(columns):
            quant_weight = quantize(weight)
            history_indices = permutation_weight[:, :step]
            weight_history = gather_columns(weight_orig, history_indices).to(first.dtype)
            quant_history = gather_columns(quant_weight, history_indices).to(first.dtype)
            lw = torch.bmm(weight_history, lg[:, step, :step].to(device).unsqueeze(2)).squeeze(2)
            lq = torch.bmm(quant_history, lh[:, step, :step].to(device).unsqueeze(2)).squeeze(2)
            current_indices = permutation_weight[:, step]
            current_weight = gather_columns(weight,
                                            current_indices[:, None]).squeeze(2).to(first.dtype)
            q_arg = ds[:, step].to(device).unsqueeze(1) * current_weight + lw - lq
            weight = scatter_column(weight, current_indices, q_arg.to(dtype))

        return FunctionalGPxQBatch.writeback(targets, weight)


class gpfq_mode(gpxq_mode):
    """
    Apply GPFQ algorithm, or other algorithms that solve the mismatched objective function,
    like Qronos or A2GPFQ.

    Args:
        model (Module): The model to quantize with GPFQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is
            a group of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPFQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before
            applying gpfq. These weights will be used anytime quantization is disabled.
            Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while
            performing GPFQ. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise
            the forward call inside the context manager returns None. Default: False
        act_order (bool): Whether to order greedy path following by Hessian approximation.
            Default: False
        algorithm_impl (GPFQ): The uninitialized class to execute the algorithm.
            Default: `brevitas.graph.gpfq.GPFQ`
        device (str): Device the buffers are stored on. Default: cpu
        dtype (torch.dtype): Datatype the buffers are stored in. Default: torch.float32

    Example:
        >>> with torch.no_grad():
        >>>     with gpfq_mode(model) as gpfq:
        >>>         gpfq_model = gpfq.model
        >>>         for i in tqdm(range(gpfq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gpfq_model(img)
        >>>             gpfq.update()
    """

    def __init__(
            self,
            model: nn.Module,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            return_forward_output: bool = False,
            act_order: bool = False,
            algorithm_impl: GPFQ = GPFQ,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32,
            functional_state=None,
            min_samples: int = 0,
            insufficient_samples: str = 'rtn',
            expert_batch_size: int = 1) -> None:
        super().__init__(
            model,
            group_of_parallel_layers,
            inplace,
            create_weight_orig,
            use_quant_activations,
            act_order,
            return_forward_output,
            device,
            dtype,
            functional_state,
            min_samples,
            insufficient_samples,
            expert_batch_size)

        self.algorithm_impl = algorithm_impl
        self._routing_phase = None
        self._routing_cache = {}
        self._routing_replay_index = {}
        self._routing_hook_handles = []

    def __enter__(self):
        mode = super().__enter__()
        if self.functional_state is not None:
            owner_modules = {target.owner.owner for target in self.functional_targets}
            for module in owner_modules:
                self._routing_hook_handles.append(
                    module.register_forward_pre_hook(self._routing_hook, with_kwargs=True))
        return mode

    def __exit__(self, type, value, traceback):
        for handle in self._routing_hook_handles:
            handle.remove()
        self._routing_hook_handles.clear()
        self._routing_phase = None
        self._routing_cache.clear()
        self._routing_replay_index.clear()
        return super().__exit__(type, value, traceback)

    def _routing_hook(self, module, args, kwargs):
        """Replay quantized-pass expert assignments during the paired reference pass."""
        route_location = None
        route = None
        for name in ('selected_experts', 'top_k_index', 'expert_index'):
            candidate = kwargs.get(name)
            if isinstance(candidate, Tensor):
                route_location = name
                route = candidate
                break
        if route is None and len(args) > 1 and isinstance(args[1], Tensor):
            route_location = 1
            route = args[1]
        if route is None or route.dim() == 0 or route.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return args, kwargs

        key = id(module)
        if self._routing_phase == 'capture':
            self._routing_cache.setdefault(key, []).append(route.detach().clone())
        elif self._routing_phase == 'replay':
            index = self._routing_replay_index.get(key, 0)
            cached_routes = self._routing_cache.get(key, ())
            if index >= len(cached_routes):
                raise RuntimeError('Functional GPxQ reference pass has no matching expert route.')
            replay_route = cached_routes[index].to(route.device)
            if replay_route.shape != route.shape:
                raise RuntimeError(
                    'Functional GPxQ expert routing shape changed between paired passes.')
            self._routing_replay_index[key] = index + 1
            if isinstance(route_location, str):
                kwargs = dict(kwargs)
                kwargs[route_location] = replay_route
            else:
                args = list(args)
                args[route_location] = replay_route
                args = tuple(args)
        return args, kwargs

    def _update_functional_targets(self, targets, progress) -> int:
        algorithm_class = self.algorithm_impl.func if isinstance(
            self.algorithm_impl, partial) else self.algorithm_impl
        batch_impl = getattr(algorithm_class, 'batched_layer_update', None)
        if self.expert_batch_size == 1 or batch_impl is None:
            return super()._update_functional_targets(targets, progress)
        failed = []
        for start in range(0, len(targets), self.expert_batch_size):
            batch_targets = targets[start:start + self.expert_batch_size]
            optimizers = [self.gpxq_layers[target.name] for target in batch_targets]
            failed.extend(batch_impl(optimizers))
            progress.set_postfix(batch=len(batch_targets), failed=len(failed))
            progress.update(len(batch_targets))
        return len(failed)

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        self._routing_cache.clear()
        self._routing_replay_index.clear()
        self._routing_phase = 'capture'
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Disable quantization
        # TODO: Ensure that removing is_training=False does not cause any regression and remove,
        # if that is the case
        targets = self.functional_targets if self.functional_state is not None else ()
        if self.functional_state is not None:
            self.functional_state.reset_active_counters()
        for target in targets:
            target.reference_pass = True
        self._routing_phase = 'replay'
        try:
            with quantization_status_manager(
                    self.model,
                    disable_act_quant=True,
                    disable_weight_quant=True,
                    disable_bias_quant=True,
                    is_training=False,
            ):
                try:
                    self.orig_forward(*args, **kwargs)
                except StopFwdException:
                    pass
        finally:
            self._routing_phase = None
            for target in targets:
                target.reference_pass = False

        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            out = self.orig_forward(*args, **kwargs)
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = False
            return out

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        return self.algorithm_impl(
            layer=layer,
            name=name,
            act_order=self.act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            device=self.device,
            dtype=self.dtype)
