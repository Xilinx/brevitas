# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

import warnings

from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.qronos import Qronos
from brevitas.graph.utils import is_conv_transposed


class GPTAQ(Qronos):
    """
    Implementation of GPTAQ as proposed in: https://arxiv.org/pdf/2504.02692
    """

    def single_layer_update(self, percdamp: float = 0.01):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPTAQ does not support weight quantizers that require metadata from input quantizers."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        if self.use_intermediate_buffer:
            del self.B  # free memory

        weight: Tensor = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

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

        assert not torch.isnan(self.H).any(), f"Error in {self.name}"
        assert not torch.isnan(self.G).any(), f"Error in {self.name}"

        # In their paper, dXXT = (\tilde{X} - X) X^T, where X is quantized and
        # \tilde{X} is not. We precompute G (\tilde{X}X^T) and H (XX^T) for Qronos,
        # so we can just use that here and initialize P to dXXT for GPTAQ
        self.P = self.G - self.H
        del self.G  # free memory

        self.L = self.H.clone()
        # Try/Except in case the inverse Hessian cannot be computed
        try:
            for i in range(self.groups):
                damp = percdamp * torch.mean(torch.diag(self.H[i]))
                diag = torch.arange(self.columns, device=self.device)
                self.L[i, diag, diag] += damp
                self.L[i] = torch.linalg.cholesky(self.L[i])
                self.L[i] = torch.cholesky_inverse(self.L[i])
                self.L[i] = torch.linalg.cholesky(self.L[i], upper=True)
                # calculate their P matrix and adjust with their ad-hoc scaling; this is not reported in
                # their paper, but it is used in their code, reportedly to stabilize results. They do not
                # explain where it comes from in their derivations, but they set it to 0.25 by default, 
                # presumably from light hyperparameter tuning. We expose this here since it is unclear if
                # it is sensitive to models, bit widths, datasets, etc. For the relevant PR, see
                # https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ/issues/4
                self.P[i] = self.alpha * (((self.P[i].to(dev) @ self.L[i].T.to(dev)).triu_(diagonal=1)) @ self.L[i].to(dev)).to(self.device)
        except LinAlgError as e:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTAQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            weight_block = weight[:, :, perm[i1:i2]].to(self.dtype)  # [groups, OC/groups, i2-i1]
            error_block = torch.zeros_like(
                weight_block, dtype=self.dtype)  # [groups, OC/groups, i2-i1]

            h_inv_block = self.L[:, i1:i2, i1:i2]
            p_block = self.P[:, i1:i2, i1:i2]
            for i in range(count):
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index].to(self.dtype)  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(self.dtype)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights and adjust
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(
                            h_inv_block[group_index, i, i:].unsqueeze(0).to(dev)) - \
                                w.unsqueeze(1).matmul(
                                    p_block[group_index, i, i:].unsqueeze(0).to(dev))).to(dtype)

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(
                        self.L[group_index, i1:i2, i2:].to(dev)) - \
                            weight_block[group_index].matmul(
                                self.P[group_index, i1:i2, i2:].to(dev))).to(dtype)

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)

        del self.L, self.P  # free memory
