# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from time import perf_counter
from typing import List
from typing import Optional
from typing import Sequence
import warnings

import torch

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

from brevitas.graph.functional_quant import FunctionalLinearTarget
from brevitas.graph.gpxq import FunctionalGPxQBatch
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.utils.torch_utils import StopFwdException


class GPTQ(GPxQ):
    """
    Adapted from https://github.com/IST-DASLab/gptq, released under the following LICENSE:

    Copyright 2023 IST-DASLab

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks,
            device='cpu',
            dtype=torch.float32) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, device, dtype)

        # Define how many columns to update in each mini-block
        self.blocksize = math.ceil(self.columns / num_blocks)

        # Initialize Hessian matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device=self.device,
                             dtype=self.dtype)
        if self.use_intermediate_buffer:
            # Creating an intermediate buffer with pinned memory to improve transfer speeds
            self.B = torch.zeros((self.groups, self.columns, self.columns),
                                 pin_memory=torch.cuda.is_available(),
                                 device=self.device,
                                 dtype=self.dtype)
        self.nsamples = 0

    def compute_iterative_covariance(self, module, input, current_layer):
        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        # NOTE: batch_size = seqlen for language models here
        inp_processed = self.process_input(input)  # [groups, in_features, batch_size]
        batch_size = inp_processed.shape[-1]
        # Calcuate the covariance matrix
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.to(self.dtype)
        if self.use_intermediate_buffer:
            # Optimizing CPU to GPU transfer using in-place copy to pinned memory
            self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
            self.H += self.B
        else:
            self.H += inp_processed.bmm(inp_processed.transpose(2, 1))

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input
        # Workaround to avoid duplication with GPTQ and MagR, will have the same method
        # across GPxQ classes
        self.compute_iterative_covariance(module, input, current_layer)
        # If we are executing GPTQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            if current_layer.stop_forward:
                raise StopFwdException

    def single_layer_update(self, percdamp=.01, c=1e4):
        assert not self.layer.weight_quant.requires_quant_input, "Error: GPTQ does not support weight quantizers that require quantized inputs."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        if self.use_intermediate_buffer:
            del self.B  # free memory
        weight = self.layer.weight.data
        functional_weight_before = (
            weight.detach().clone() if isinstance(self.layer, FunctionalLinearTarget) else None)
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # List with permutation tensors for the Hessian and Weight matrix.
        # If act_order is False, the tensors will be ordered indexes.
        # For groupwise convolution, we have one tensor per group,
        # thus len(permutation_list) is always equal to self.groups.
        # We do not explicity permute the weight matrix, only the Hessian.
        permutation_list = []
        weight = weight.view(self.groups, -1, weight.shape[-1])
        # For groupwise convolution, these operations are groupwise so we iterate
        for i in range(self.groups):
            # If a diagonal element on the Hessian is zero, we can set to 0 the corresponding
            # column in the weight matrix.
            # The diagonal element is set to 1 to avoid division-by-zero
            dead = torch.diag(self.H[i, :, :]) == 0
            self.H[i, dead, dead] = 1
            # If the diagonal of activations is zero, we set the weight to zero
            weight[i, :, dead] = 0
            if self.act_order:
                # Re-order Hessian so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(torch.diag(self.H[i, :, :]), descending=True)
                self.H[i, :, :] = self.H[i, perm, :][:, perm]
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(self.H.shape[-1]), device=dev)
            permutation_list.append(perm)

        # Try/Except in case the inverse Hessian cannot be computed
        try:
            for i in range(self.groups):
                damp = percdamp * torch.mean(torch.diag(self.H[i, :, :]))
                diag = torch.arange(self.columns, device=self.device)
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                # stabilizing the Cholesky decomposition with a fairly large constant, c
                self.H[i, :, :] = torch.linalg.cholesky(
                    self.H[i, :, :] * c, upper=True) / math.sqrt(c)
            h_inv = self.H
        except LinAlgError as e:
            if functional_weight_before is not None:
                self.layer.writeback(functional_weight_before)
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return True
        finally:
            del self.H

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=self.dtype)  # [groups, OC/groups, i2-i1]

            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index].to(self.dtype)  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(self.dtype)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(
                            h_inv_block[group_index, i, i:].unsqueeze(0).to(dev))).to(dtype)

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
        if functional_weight_before is not None and not torch.isfinite(self.layer.weight).all():
            self.layer.writeback(functional_weight_before)
            warnings.warn(
                f'GPTQ update for layer {self.name} produced non-finite weights; '
                'restoring its pre-GPTQ weights.')
            return True
        return False

    @staticmethod
    def batched_layer_update(optimizers: Sequence['GPTQ'], percdamp=.01, c=1e4):
        """Update compatible functional expert matrices with batched tensor operations."""
        batch = FunctionalGPxQBatch(optimizers)
        first = optimizers[0]
        columns = first.columns
        blocksize = first.blocksize
        act_order = first.act_order
        if any(optimizer.columns != columns or optimizer.blocksize != blocksize or
               optimizer.act_order != act_order for optimizer in optimizers):
            raise ValueError('Batched functional GPTQ requires compatible expert optimizers.')
        for optimizer in optimizers:
            if optimizer.layer.weight_quant.requires_quant_input:
                raise RuntimeError(
                    'GPTQ does not support weight quantizers that require quantized inputs.')
            if optimizer.use_intermediate_buffer:
                del optimizer.B

        targets = batch.targets
        weight = batch.weight
        weight_device = weight.device
        weight_dtype = weight.dtype
        hessian = batch.pop_buffer('H', first.device)

        diagonal = hessian.diagonal(dim1=-2, dim2=-1)
        dead = diagonal == 0
        diagonal[dead] = 1
        weight.masked_fill_(dead.to(weight_device).unsqueeze(1), 0)
        if act_order:
            permutation = torch.argsort(diagonal, dim=-1, descending=True)
            hessian = torch.gather(hessian, 1, permutation.unsqueeze(-1).expand(-1, -1, columns))
            hessian = torch.gather(hessian, 2, permutation.unsqueeze(1).expand(-1, columns, -1))
        else:
            permutation = torch.arange(columns, device=hessian.device).expand(len(optimizers), -1)

        diagonal_index = torch.arange(columns, device=hessian.device)
        damp = percdamp * hessian.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        hessian[:, diagonal_index, diagonal_index] += damp.unsqueeze(1)
        chol, info = torch.linalg.cholesky_ex(hessian, check_errors=False)
        del hessian
        valid = info == 0
        failed = [targets[index] for index in torch.where(~valid)[0].tolist()]
        if not valid.any():
            for target in failed:
                warnings.warn(
                    f'Failed to compute the inverse Hessian for layer {target.name}; GPTQ will not be applied.'
                )
            return failed

        valid_indices = torch.where(valid)[0]
        targets = [targets[index] for index in valid_indices.tolist()]
        weight = weight.index_select(0, valid_indices.to(weight_device))
        permutation = permutation[valid]
        h_inv = torch.cholesky_inverse(chol[valid])
        del chol
        inverse_finite = torch.isfinite(h_inv).all(dim=-1).all(dim=-1)
        if not inverse_finite.all():
            inverse_failed_indices = torch.where(~inverse_finite)[0].tolist()
            failed.extend([targets[index] for index in inverse_failed_indices])
            targets = [target for index, target in enumerate(targets) if inverse_finite[index]]
            finite_indices = torch.where(inverse_finite)[0]
            weight = weight.index_select(0, finite_indices.to(weight_device))
            permutation = permutation[inverse_finite]
            h_inv = h_inv[inverse_finite]
        if not targets:
            for target in failed:
                warnings.warn(
                    f'Failed to compute the inverse Hessian for layer {target.name}; GPTQ will not be applied.'
                )
            return failed
        upper, upper_info = torch.linalg.cholesky_ex(h_inv * c, upper=True, check_errors=False)
        del h_inv
        second_valid = upper_info == 0
        second_failed_indices = torch.where(~second_valid)[0].tolist()
        second_failed = [targets[index] for index in second_failed_indices]
        failed.extend(second_failed)
        for target in failed:
            warnings.warn(
                f'Failed to compute the inverse Hessian for layer {target.name}; GPTQ will not be applied.'
            )
        if not second_valid.any():
            return failed

        targets = [target for index, target in enumerate(targets) if second_valid[index]]
        second_valid_indices = torch.where(second_valid)[0]
        weight = weight.index_select(0, second_valid_indices.to(weight_device))
        permutation = permutation[second_valid]
        h_inv = upper[second_valid] / math.sqrt(c)
        del upper
        permutation_weight = permutation.to(weight_device)

        def quantize_weight(value):
            return batch.quantize(targets, value)

        def gather_columns(value, indices):
            return torch.gather(value, 2, indices.unsqueeze(1).expand(-1, value.shape[1], -1))

        def scatter_columns(value, indices, update):
            return value.scatter(2, indices.unsqueeze(1).expand(-1, value.shape[1], -1), update)

        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            block_indices = permutation_weight[:, i1:i2]
            error_block = torch.zeros((len(targets), weight.shape[1], count),
                                      dtype=first.dtype,
                                      device=weight_device)
            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for index in range(count):
                quant_weight = quantize_weight(weight)
                column_indices = block_indices[:, index:index + 1]
                q = gather_columns(quant_weight, column_indices).squeeze(2).to(first.dtype)
                w = gather_columns(weight, column_indices).squeeze(2).to(first.dtype)
                divisor = h_inv_block[:, index, index].to(weight_device).unsqueeze(1)
                error = (w - q) / divisor
                error_block[:, :, index] = error
                remaining_indices = block_indices[:, index:]
                remaining = gather_columns(weight, remaining_indices)
                h_row = h_inv_block[:, index, index:].to(weight_device)
                remaining -= (error.unsqueeze(2) * h_row.unsqueeze(1)).to(weight_dtype)
                weight = scatter_columns(weight, remaining_indices, remaining)

            if i2 < columns:
                tail_indices = permutation_weight[:, i2:]
                tail = gather_columns(weight, tail_indices)
                correction = torch.bmm(error_block, h_inv[:, i1:i2,
                                                          i2:].to(weight_device)).to(weight_dtype)
                weight = scatter_columns(weight, tail_indices, tail - correction)

        failed.extend(FunctionalGPxQBatch.writeback(targets, weight))
        return failed


class gptq_mode(gpxq_mode):
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323.

    Args:
        model (Module): The model to quantize with GPTQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPTQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gptq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPTQ. Default: False
        num_blocks (int): The number of sub-blocks to use to speed-up GPTQ computation. Default: 100
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False
        gptq_class (GPTQ): The uninitialized class to perform GPTQ. Default: `brevitas.graph.gptq.GPTQ`
        device (str): Device the buffers are stored on. Default: cpu
        dtype (torch.dtype): Datatype the buffers are stored in. Default: torch.float32

    Example:
        >>> with torch.no_grad():
        >>>     with gptq_mode(model) as gptq:
        >>>         gptq_model = gptq.model
        >>>         for i in tqdm(range(gptq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gptq_model(img)
        >>>             gptq.update()
    """

    def __init__(
            self,
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            num_blocks: int = 100,
            return_forward_output: bool = False,
            act_order: bool = False,
            gptq_class: GPTQ = GPTQ,
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

        # How many subblock to use during GPTQ for each layer
        self.num_blocks = num_blocks
        self.gptq_class = gptq_class

    def _update_functional_targets(self, targets, progress) -> int:
        if self.expert_batch_size == 1:
            failed = 0
            for target in targets:
                optimizer = self.gpxq_layers[target.name]
                target_start = perf_counter()
                failed += int(optimizer.single_layer_update() is True)
                progress.set_postfix(
                    samples=optimizer.nsamples, seconds=f'{perf_counter() - target_start:.1f}')
                progress.update()
            return failed
        failed = []
        for start in range(0, len(targets), self.expert_batch_size):
            batch_targets = targets[start:start + self.expert_batch_size]
            optimizers = [self.gpxq_layers[target.name] for target in batch_targets]
            failed.extend(self.gptq_class.batched_layer_update(optimizers))
            progress.set_postfix(batch=len(batch_targets), failed=len(failed))
            progress.update(len(batch_targets))
        return len(failed)

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        finally:
            if self.return_forward_output:
                # If we want to return the output of the network, we need to disable all hooks
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = True
                out = self.orig_forward(*args, **kwargs)
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = False
                return out

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        return self.gptq_class(
            layer=layer,
            name=name,
            act_order=self.act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            num_blocks=self.num_blocks,
            device=self.device,
            dtype=self.dtype)
