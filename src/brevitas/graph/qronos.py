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

from brevitas.graph.functional_quant import FunctionalLinearTargetBatch
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

        is_quant_enabled = not module.reference_pass if hasattr(
            module, 'reference_pass') else module.weight_quant.is_quant_enabled

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
            if current_layer.stop_forward:
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
            del self.iH, self.G, self.H
            return

        self.iH = self.iH.to(dev)
        self.G = self.G.to(dev)
        self.H = self.H.to(dev)

        dtype_min = torch.finfo(dtype).min
        dtype_max = torch.finfo(dtype).max

        # Qronos - step 1
        q_groups = self.get_quant_weights(0, 0, permutation_list, with_quant_history=True)
        for group_index in range(self.groups):
            perm = permutation_list[group_index]
            q: Tensor = q_groups[group_index].to(self.dtype)
            v: Tensor = weight[group_index, :, perm].to(self.dtype)
            w: Tensor = weight_orig[group_index, :, perm].to(self.dtype)
            Gw = w.matmul(self.G[group_index, :, 0] * Dhi[group_index, 0])
            Uv = v.matmul(Uh[group_index, 0, :] * Dhi[group_index, 0])
            q_arg = Gw - Uv
            assert (q_arg >= dtype_min).all() and (q_arg <= dtype_max).all()
            weight[group_index, :, perm[0]] = q_arg.to(dtype)

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
            del self.L, self.iH
            return
        del self.iH  # memory management

        # Qronos - step 2+
        for i1 in range(1, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=self.dtype)  # [groups, OC/groups, i2-i1]
            # we need to decrement once because of the Sherman-Morrison-Woodbury update
            h_inv_block = self.L[:, i1 - 1:i2 - 1, i1 - 1:i2 - 1]
            # correct error within the block
            for i in range(count):
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

    @staticmethod
    def batched_layer_update(optimizers, beta: int = 1e4):
        """Apply Qronos to a compatible batch of functional expert matrices."""
        if not optimizers:
            return []
        first = optimizers[0]
        columns = first.columns
        blocksize = first.blocksize
        if any(optimizer.groups != 1 or optimizer.columns != columns or
               optimizer.blocksize != blocksize or optimizer.act_order != first.act_order or
               optimizer.alpha != first.alpha or
               optimizer.layer.weight.shape != first.layer.weight.shape
               for optimizer in optimizers):
            raise ValueError('Batched functional Qronos requires compatible expert optimizers.')
        for optimizer in optimizers:
            assert not optimizer.layer.weight_quant.requires_quant_input, \
                'Error: Qronos does not support weight quantizers that require metadata from input quantizers.'
            assert optimizer.quant_input is None, 'Qronos quantized and reference inputs are unbalanced.'
            if optimizer.use_intermediate_buffer and hasattr(optimizer, 'B'):
                del optimizer.B

        targets = [optimizer.layer for optimizer in optimizers]
        weight = torch.stack([target.weight.detach() for target in targets])
        weight_orig = torch.stack([target.weight_orig.to(weight.device) for target in targets])
        device = weight.device
        dtype = weight.dtype
        hessian = torch.cat([optimizer.H for optimizer in optimizers], dim=0)
        cross_covariance = torch.cat([optimizer.G for optimizer in optimizers], dim=0)
        for optimizer in optimizers:
            del optimizer.H, optimizer.G

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
        if not torch.isfinite(hessian).all() or not torch.isfinite(cross_covariance).all():
            raise RuntimeError('Batched functional Qronos received non-finite covariance matrices.')

        diagonal_h = hessian.diagonal(dim1=-2, dim2=-1)
        reciprocal_h = torch.where(diagonal_h != 0, 1. / diagonal_h, torch.zeros_like(diagonal_h))
        upper_h = torch.triu(hessian, diagonal=1)

        scale = hessian.amax(dim=(-2, -1)).abs()
        valid_scale = scale > 0
        normalized = hessian / torch.where(valid_scale, scale,
                                           torch.ones_like(scale))[:, None, None]
        generator = torch.Generator(device=hessian.device).manual_seed(42)
        vector = torch.rand(
            columns, device=hessian.device, dtype=first.dtype,
            generator=generator).expand(len(optimizers), -1).clone()
        for _ in range(30):
            next_vector = torch.bmm(normalized, vector.unsqueeze(2)).squeeze(2)
            vector = next_vector / (
                torch.linalg.vector_norm(next_vector, dim=1, keepdim=True) + 1e-12)
        singular_value = torch.bmm(vector.unsqueeze(1), torch.bmm(
            normalized, vector.unsqueeze(2))).flatten() * scale
        damp = first.alpha * singular_value
        inverse_input = hessian.clone()
        diag_index = torch.arange(columns, device=hessian.device)
        inverse_input[:, diag_index, diag_index] += damp.unsqueeze(1)
        chol, info = torch.linalg.cholesky_ex(inverse_input, check_errors=False)
        valid = (info == 0) & valid_scale & torch.isfinite(damp)
        failed = [targets[index] for index in torch.where(~valid)[0].tolist()]
        if not valid.any():
            return failed

        valid_indices = torch.where(valid)[0]
        targets = [targets[index] for index in valid_indices.tolist()]
        weight = weight.index_select(0, valid_indices.to(device))
        weight_orig = weight_orig.index_select(0, valid_indices.to(device))
        hessian = hessian[valid]
        cross_covariance = cross_covariance[valid]
        reciprocal_h = reciprocal_h[valid]
        upper_h = upper_h[valid]
        permutation = permutation[valid]
        damp = damp[valid]
        try:
            inverse_h = torch.cholesky_inverse(chol[valid])
        except RuntimeError:
            return failed + targets

        permutation_weight = permutation.to(device)

        def gather_columns(value, indices):
            return torch.gather(value, 2, indices.unsqueeze(1).expand(-1, value.shape[1], -1))

        def scatter_columns(value, indices, update):
            return value.scatter(2, indices.unsqueeze(1).expand(-1, value.shape[1], -1), update)

        ordered_weight = gather_columns(weight, permutation_weight).to(first.dtype)
        ordered_weight_orig = gather_columns(weight_orig, permutation_weight).to(first.dtype)
        gw = torch.bmm(
            ordered_weight_orig,
            (cross_covariance[:, :, 0] *
             reciprocal_h[:, 0].unsqueeze(1)).to(device).unsqueeze(2)).squeeze(2)
        uv = torch.bmm(
            ordered_weight, (upper_h[:, 0, :] *
                             reciprocal_h[:, 0].unsqueeze(1)).to(device).unsqueeze(2)).squeeze(2)
        ordered_weight[:, :, 0] = gw - uv

        a_matrix = inverse_h[:, 1:, 1:]
        b_vector = inverse_h[:, 1:, 0]
        a_matrix = a_matrix - b_vector.unsqueeze(2) * b_vector.unsqueeze(
            1) / inverse_h[:, 0, 0][:, None, None]
        inverse_h = a_matrix

        weight = scatter_columns(weight, permutation_weight, ordered_weight.to(dtype))
        quant_weight = torch.stack([
            target.quantize(weight[index]) for index, target in enumerate(targets)])
        if all(getattr(target.owner.proxy, 'is_groupwise', False) for target in targets):
            try:
                batch_quant = FunctionalLinearTargetBatch(targets, weight)
                candidate = batch_quant.quant_weight(weight)
                if torch.equal(candidate, quant_weight):
                    quant_weight = candidate
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        quant_history = gather_columns(quant_weight, permutation_weight[:, :1]).to(first.dtype)
        identity = torch.diag_embed(damp[:, None].expand(-1, columns)).to(device)
        gh = cross_covariance.to(device) + identity
        gw = torch.bmm(ordered_weight_orig, torch.bmm(gh[:, :, 1:], inverse_h.to(device)))
        hq = torch.bmm(
            quant_history, torch.bmm(hessian[:, :1, 1:].to(device), inverse_h.to(device)))
        ordered_weight[:, :, 1:] = gw - hq
        weight = scatter_columns(weight, permutation_weight, ordered_weight.to(dtype))
        del cross_covariance, hessian

        upper, upper_info = torch.linalg.cholesky_ex(
            inverse_h * beta, upper=True, check_errors=False)
        second_valid = upper_info == 0
        failed.extend([targets[index] for index in torch.where(~second_valid)[0].tolist()])
        if not second_valid.any():
            return failed
        second_indices = torch.where(second_valid)[0]
        targets = [targets[index] for index in second_indices.tolist()]
        weight = weight.index_select(0, second_indices.to(device))
        permutation_weight = permutation_weight.index_select(0, second_indices.to(device))
        l_factor = upper[second_valid] / math.sqrt(beta)

        batch_quant = None
        use_batch_quant = False
        if all(getattr(target.owner.proxy, 'is_groupwise', False) for target in targets):
            try:
                batch_quant = FunctionalLinearTargetBatch(targets, weight)
                use_batch_quant = torch.equal(
                    batch_quant.quant_weight(weight),
                    torch.stack([
                        target.quantize(weight[index]) for index, target in enumerate(targets)]))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                batch_quant = None

        def quantize(value):
            if use_batch_quant:
                return batch_quant.quant_weight(value)
            return torch.stack([
                target.quantize(value[index]) for index, target in enumerate(targets)])

        for i1 in range(1, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            block_indices = permutation_weight[:, i1:i2]
            error_block = torch.zeros((len(targets), weight.shape[1], count),
                                      dtype=first.dtype,
                                      device=device)
            inverse_block = l_factor[:, i1 - 1:i2 - 1, i1 - 1:i2 - 1]
            for index in range(count):
                quant_weight = quantize(weight)
                column_indices = block_indices[:, index:index + 1]
                q = gather_columns(quant_weight, column_indices).squeeze(2).to(first.dtype)
                w = gather_columns(weight, column_indices).squeeze(2).to(first.dtype)
                divisor = inverse_block[:, index, index].to(device).unsqueeze(1)
                error = (w - q) / divisor
                error_block[:, :, index] = error
                remaining_indices = block_indices[:, index:]
                remaining = gather_columns(weight, remaining_indices)
                inverse_row = inverse_block[:, index, index:].to(device)
                remaining -= (error.unsqueeze(2) * inverse_row.unsqueeze(1)).to(dtype)
                weight = scatter_columns(weight, remaining_indices, remaining)
            if i2 < columns:
                tail_indices = permutation_weight[:, i2:]
                tail = gather_columns(weight, tail_indices)
                correction = torch.bmm(error_block, l_factor[:, i1 - 1:i2 - 1,
                                                             i2 - 1:].to(device)).to(dtype)
                weight = scatter_columns(weight, tail_indices, tail - correction)

        for index, target in enumerate(targets):
            target.writeback(weight[index])
        return failed

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
