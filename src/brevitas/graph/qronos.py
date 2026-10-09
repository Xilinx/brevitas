# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List
from typing import Tuple

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

        # Mixins may override _resolve_blocksize (e.g. PiSO aligns blocks with groups).
        self.blocksize = self._resolve_blocksize(num_blocks)

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

    def _step2_block_ranges(self) -> List[Tuple[int, int]]:
        """(i1, i2) column blocks for the Qronos step-2+ error-correction loop.

        Column 0 is handled in step 1, so blocks start at column 1. This is the
        reference Qronos partition: range(1, columns, blocksize).

        This lives in a hook because the group-interleaved PiSO variant needs a
        different partition (each block must cover exactly one quantization group
        so a group's scale can be optimized at its boundary). Keeping the base
        partition here guarantees that plain Qronos and layer-interleaved PiSO
        reproduce the reference numerics exactly: block boundaries determine how
        quantization error is diffused, so any change to them changes the result
        even without scale optimization. Only PiSOGroupInterleavedQronos overrides
        this (see _PiSOQronosMixin).
        """
        return [(i1, min(i1 + self.blocksize, self.columns))
                for i1 in range(1, self.columns, self.blocksize)]

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
            if self.act_order:
                # Re-order covariance matrices so that weights associated to higher
                # magnitude activations are quantized first. Mixins may override
                # _act_order_permutation (e.g. group-aware ordering).
                perm = self._act_order_permutation(self.H[group_index])
                self.G[group_index] = self.G[group_index, perm, :][:, perm]
                self.H[group_index] = self.H[group_index, perm, :][:, perm]
            else:
                # No permutation, permutation tensor is an ordered index
                perm = torch.tensor(range(self.H.shape[-1]), device=dev)
            perm = perm.to(weight.device)
            permutation_list.append(perm)

        assert not torch.isnan(self.H).any(), f"Error in {self.name}"
        assert not torch.isnan(self.G).any(), f"Error in {self.name}"

        Dh: Tensor = torch.zeros((self.groups, self.columns), device=self.device, dtype=self.dtype)
        for group_index in range(self.groups):
            Dh[group_index].copy_(self.H[group_index].diag())
        Dhi = torch.where(Dh != 0, 1. / Dh, torch.zeros_like(Dh)).to(dev)  # D^{-1}

        Uh: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                 device=dev,
                                 dtype=self.dtype)
        for group_index in range(self.groups):
            Uh[group_index].copy_(torch.triu(self.H[group_index], 1))  # upper (for future)

        # Try/Except in case the inverse cannot be computed
        self.iH = self.H.clone()
        diag = torch.arange(self.columns, device=self.device)
        damp = torch.zeros(self.groups, device=self.device)
        try:
            for group_index in range(self.groups):
                # using power iteration to estimate the maximum singular value
                damp[group_index] = self.alpha * power_iteration(self.H[group_index], 30)
                self.iH[group_index, diag, diag] += damp[group_index]
                self.iH[group_index] = torch.linalg.cholesky(self.iH[group_index])
                self.iH[group_index] = torch.cholesky_inverse(self.iH[group_index])
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of H for layer {self.name} '
                f'Forward error correction will be a null operation. '
                f'Increasing the number of samples might fix this issue.')
            self._on_layer_finished()
            if hasattr(self.layer, 'offload_params'):
                self.layer.offload_params(self.layer)
            return

        self.iH = self.iH.to(dev)
        self.G = self.G.to(dev)
        self.H = self.H.to(dev)

        # Hook before error correction (PiSO optimizes the layer scale here);
        # no-op otherwise.
        self._on_error_correction_starts(perm)

        dtype_min = torch.finfo(dtype).min
        dtype_max = torch.finfo(dtype).max

        # Qronos - step 1
        for group_index in range(self.groups):
            perm = permutation_list[group_index]
            v: Tensor = weight[group_index, :, perm].to(self.dtype)
            w: Tensor = weight_orig[group_index, :, perm].to(self.dtype)
            Gw = w.matmul(self.G[group_index, :, 0] * Dhi[group_index, 0])
            Uv = v.matmul(Uh[group_index, 0, :] * Dhi[group_index, 0])
            q_arg = Gw - Uv
            assert (q_arg >= dtype_min).all() and (q_arg <= dtype_max).all()
            weight[group_index, :, perm[0]] = q_arg.to(dtype)  # weight 0 changes here

        # Optimize scale for group 0 using the Qronos-corrected col 0 value, before
        # quantization. No-op unless a PiSO mixin is present.
        self._on_column_reached(0)

        # Sherman-Morrison-Woodbury update rule
        A = self.iH[:, 1:, 1:]
        for group_index in range(self.groups):
            c = self.iH[group_index, 0, 0]
            b = self.iH[group_index, 1:, [0]]
            A[group_index] -= (b.matmul(b.T)) / c
        self.iH = A

        q_groups = self.get_quant_weights(0, 1, permutation_list, with_quant_history=True)
        for group_index in range(self.groups):
            perm = permutation_list[group_index]
            q: Tensor = q_groups[group_index].to(self.dtype)
            w: Tensor = weight_orig[group_index, :, perm].to(self.dtype)
            Ih = torch.diag(torch.full((self.columns,), damp[group_index], device=dev))
            Gh = self.G[group_index] + Ih
            Gw = w.matmul(Gh[:, 1:] @ self.iH[group_index])
            Hq = q.matmul(self.H[group_index, :1, 1:] @ self.iH[group_index])
            # all the weights changed in the following line, does it make sense to re-compute the scales here??
            weight[group_index, :, perm[1:]] = (Gw - Hq).to(dtype)

        del self.G, self.H  # memory management

        self.L = self.iH.clone()
        try:
            for group_index in range(self.groups):
                # stabilizing the Cholesky decomposition with a fairly large constant, beta
                self.L[group_index] = torch.linalg.cholesky(
                    self.iH[group_index] * beta, upper=True) / math.sqrt(beta)
        except LinAlgError:
            warnings.warn(
                f'Failed to compute Cholesky decomposition for layer {self.name} '
                f'Forward error correction will be a null operation. '
                f'Increasing the number of samples might fix this issue.')
            self._on_layer_finished()
            if hasattr(self.layer, 'offload_params'):
                self.layer.offload_params(self.layer)
            return
        del self.iH  # memory management

        # Qronos - step 2+
        for i1, i2 in self._step2_block_ranges():
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=self.dtype)  # [groups, OC/groups, i2-i1]
            # we need to decrement once because of the Sherman-Morrison-Woodbury update
            h_inv_block = self.L[:, i1 - 1:i2 - 1, i1 - 1:i2 - 1]
            # correct error within the block
            for i in range(count):
                # Let a PiSO mixin optimize a group's scale when column i1+i starts
                # a new group, before its weights are quantized. No-op otherwise.
                self._on_column_reached(i1 + i)
                # error diffusion

                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index].to(self.dtype)  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(self.dtype)  # [OC/groups]
                    d = h_inv_block[group_index, i, i].to(self.dtype)  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # update the weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(h_inv_block[group_index, i,
                                                              i:].unsqueeze(0))).to(dtype)
            # correct error outside the block
            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(self.L[group_index, i1 - 1:i2 - 1,
                                                           i2 - 1:])).to(dtype)
        del self.L  # memory management

        self._on_layer_finished()

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
