# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from torch import Tensor

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

import warnings

from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import power_iteration
from brevitas.utils.torch_utils import StopFwdException


class Qronos(GPFQ):
    """
    Implementation of Qronos as proposed in: https://openreview.net/pdf?id=7axclBCYul

    The layer update follows Remark 5.2 in `Provable Post-Training Quantization:
    Theoretical Analysis of OPTQ and Qronos` (https://arxiv.org/abs/2508.04853).
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks: int = 100,
            alpha: float = 1e-6,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, device, dtype)
        self.blocksize = math.ceil(self.columns / num_blocks)
        self.alpha = alpha

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        # NOTE: batch_size = seqlen for language models here
        inp_processed = self.process_input(input)  # [groups, in_features, batch_size]
        inp_processed = inp_processed.to(self.dtype)
        batch_size = inp_processed.shape[-1]

        is_quant_enabled = module.weight_quant.is_quant_enabled

        # NOTE: in the gpfq_mode context manager (which we use for this), we first
        # collect quant inputs, then we collect float inputs for the same batch. We
        # assume this pattern here, but will add a check just in case.

        # if quant is not enabled, then it is the float input; if it is a float input
        # then a quant input has already happened and we can update G
        if not is_quant_enabled:
            # Computing the normalized G matrix
            self.G *= (self.nsamples - batch_size) / self.nsamples
            inp_processed = inp_processed / math.sqrt(
                self.nsamples)  # NOTE: quant_input is normalized before, in the H update
            if self.use_intermediate_buffer:
                self.B.copy_(inp_processed.bmm(self.quant_input.transpose(2, 1)))
                self.G += self.B
            else:
                self.G += inp_processed.bmm(self.quant_input.transpose(2, 1))
            self.quant_input = None  # NOTE: set back to None now that we've used it
        else:
            # Computing the normalized H matrix
            self.nsamples += batch_size  # NOTE: only increment with quant inputs
            self.H *= (self.nsamples - batch_size) / self.nsamples
            inp_processed = inp_processed / math.sqrt(self.nsamples)
            if self.use_intermediate_buffer:
                self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
                self.H += self.B
            else:
                self.H += inp_processed.bmm(inp_processed.transpose(2, 1))
            # store the quantized input for computing the H matrix
            assert self.quant_input is None
            self.quant_input = inp_processed

        # If we are executing Qronos with group_of_parallel_layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self, beta: int = 1e4):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: Qronos does not support weight quantizers that require metadata from input quantizers."
        assert hasattr(self.layer, 'weight_orig'), \
            "Error: Qronos requires the original weights to be stored, see `create_weight_orig`."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        if self.use_intermediate_buffer:
            del self.B  # free memory

        weight: Tensor = self.layer.weight.data
        weight_orig: Tensor = self.layer.weight_orig.data
        dev = weight.device
        weight_orig = weight_orig.to(dev)

        # Store the original dtype of the weights.
        # Convert computations to self.dtype and cast weight updates back to this dtype.
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

        assert not torch.isnan(self.H).any(), f"Error in {self.name}"
        assert not torch.isnan(self.G).any(), f"Error in {self.name}"

        # Compute the regularized inverse once for the projection and GPTQ.
        self.iH = self.H.clone()
        diag = torch.arange(self.columns, device=self.device)
        damp = torch.zeros(self.groups, device=self.device, dtype=self.dtype)

        # Try to compute the inverse of the regularized Hessian.
        try:
            for group_index in range(self.groups):
                # Estimate the maximum eigenvalue with power iteration.
                damp[group_index] = self.alpha * power_iteration(self.H[group_index], 30)
                self.iH[group_index, diag, diag] += damp[group_index]
                self.iH[group_index] = torch.linalg.cholesky(self.iH[group_index])
                self.iH[group_index] = torch.cholesky_inverse(self.iH[group_index])
                # Apply the same ridge term to the cross-covariance.
                self.G[group_index, diag, diag] += damp[group_index]
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of H for layer {self.name}. '
                f'Qronos will not be applied. '
                f'Increasing the number of samples might fix this issue.')
            return

        self.iH = self.iH.to(dev)
        self.G = self.G.to(dev)

        # Apply the regularized least-squares projection from Remark 5.2.
        for group_index in range(self.groups):
            weight[group_index].copy_((
                weight_orig[group_index].to(self.dtype).matmul(self.G[group_index]).matmul(
                    self.iH[group_index])).to(dtype))

        # List the permutations for the inverse Hessian and weight matrix.
        # Keep one permutation for each convolution group.
        permutation_list = []
        for group_index in range(self.groups):
            if self.act_order:
                # Quantize weights with larger activation magnitudes first.
                perm = torch.argsort(self.H[group_index].diag(), descending=True)
            else:
                # Keep the original order.
                perm = torch.arange(self.columns, device=dev)
            permutation_list.append(perm.to(dev))
            self.iH[group_index] = self.iH[group_index, perm, :][:, perm]

        del self.G, self.H  # memory management

        # Compute the upper Cholesky factor used by GPTQ error diffusion.
        try:
            for group_index in range(self.groups):
                # Scale the inverse before factorization for numerical stability.
                self.iH[group_index] = torch.linalg.cholesky(
                    self.iH[group_index] * beta, upper=True) / math.sqrt(beta)
        except LinAlgError:
            warnings.warn(
                f'Failed to compute Cholesky decomposition for layer {self.name}. '
                f'Qronos will not be applied. '
                f'Increasing the number of samples might fix this issue.')
            return

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, :count], dtype=self.dtype)  # [groups, OC/groups, i2-i1]
            h_inv_block = self.iH[:, i1:i2, i1:i2]

            # Correct the quantization error within the block.
            for i in range(count):
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index].to(self.dtype)  # [OC/groups]
                    w = weight[group_index, :, perm[i1 + i]].to(self.dtype)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # Update the remaining weights in the block.
                    weight[group_index, :, perm[i1 + i:i2]] -= (
                        error.unsqueeze(1).matmul(h_inv_block[group_index, i,
                                                              i:].unsqueeze(0))).to(dtype)

            # Correct the quantization error outside the block.
            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(self.iH[group_index, i1:i2, i2:])).to(dtype)
        del self.iH  # memory management

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
