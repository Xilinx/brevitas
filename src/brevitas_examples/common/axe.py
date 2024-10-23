# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import numpy as np
import torch
from torch import Tensor

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

from brevitas.graph.gpfq import GPFQv2
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.gpxq import SUPPORTED_TCONV_OP


def _get_average_of_nonzero_magnitudes(vec: np.ndarray, radius: float = 1.0):
    assert radius > 0, "Error: radius needs to be strictly positive."
    assert vec.ndim == 1, "Error: projection assumes a vector, not a matrix."
    assert vec.min() >= 0, "Error: assuming a vector of non-negative numbers."
    n_elems = vec.shape[0]
    # if we are already within the simplex, then the best projection is itself
    if vec.sum() <= radius:
        return 0.0
    # using algorithm detailed in "Efficient Projections onto the L1-Ball for Learning in High Dimensions"
    v = vec
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n_elems + 1) > (cumsum_u - radius))[0][-1]
    theta = float(cumsum_u[rho] - radius) / (rho + 1)
    return theta


def calc_average_nonzero_mag(weight: Tensor, lim: Tensor) -> Tensor:
    thetas = torch.zeros(weight.shape[0], device=weight.device)
    for i in range(weight.shape[0]):
        l = lim[i].item() if lim.ndim > 0 else lim.item()
        w = weight[i].cpu().detach().numpy()
        t = _get_average_of_nonzero_magnitudes(np.abs(w), l)
        thetas[i] = t
    return thetas


def pad_tensor_with_zeros(tensor: Tensor, tile_size: int) -> Tensor:
    pad_size = tile_size - (tensor.shape[1] % tile_size)
    if pad_size == tile_size:
        return tensor
    padding = torch.zeros((tensor.shape[0], pad_size), device=tensor.device)
    pad_tensor = torch.concat([tensor, padding], axis=1)
    return pad_tensor


class A2GPTQ(GPTQ):
    """
    Accumulator-aware GPTQ as proposed in https://arxiv.org/pdf/2409.17092
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks,
            max_accumulator_bit_width,
            max_accumulator_tile_size) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, num_blocks)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        if self.max_accumulator_tile_size is None:
            self.max_accumulator_tile_size = self.columns
        assert self.max_accumulator_tile_size > 2, "Error: accumulator tile size needs to be bigger than 2."
        assert self.max_accumulator_bit_width > 2, "Error: accumulator bit width needs to be bigger than 2."

    def single_layer_update(self, percdamp=0.01):
        assert not self.layer.weight_quant.requires_quant_input, "Error: GPTQ does not support weight quantizers that require quantized inputs."
        if self.quant_metadata is None:
            raise ValueError(
                "Expected self.quant_metadata to calculate accumualtor bounds, but recevied None. "
                "Make sure that either the input to the model is an IntQuantTensor or the layer has an input quant enabled. "
                "Also, check if `use_quant_activations=True` in `gptq_mode` when `max_accumulator_bit_width` is specified. "
            )
        if hasattr(self.layer, "allocate_params"):
            self.layer.allocate_params(self.layer)
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # TODO: add support for signed input activations
        if self.quant_metadata.signed:
            raise NotImplementedError("Signed inputs not yet supported.")

        n_tiles = math.ceil(weight.shape[-1] / self.max_accumulator_tile_size)
        scales: Tensor = self.layer.weight_quant.scale()
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                scales = scales.transpose(1, 0)  # This performs a view
            scales = scales.flatten(1)
        P = torch.tensor(self.max_accumulator_bit_width)
        N = self.quant_metadata.bit_width
        # NOTE: using sign-magnitude here, which is sufficient to support both
        # sign-magnitude and 2s complement accumulators
        self.upper_lim = (pow(2, P - 1) - 1) / float(pow(2, N) - 1)  # A
        self.lower_lim = -self.upper_lim  # B
        Z = (pow(2, P) - 2) / float(pow(2, N) - 1)  # l1-norm lim for zero-centered weight vector
        # translating into the quantized range; need to pad to get these thresholds
        wT = pad_tensor_with_zeros(weight / scales, self.max_accumulator_tile_size).view(
            -1, self.max_accumulator_tile_size)  # [OC * Tiles, IC / Tiles]
        thresholds = calc_average_nonzero_mag(
            wT - wT.mean(axis=1, keepdim=True), Z)  # [Groups * OC * Tiles]
        thresholds = thresholds.view(self.groups, -1,
                                     n_tiles).transpose(1, 2)  # [Groups, Tiles, OC/Groups]
        del wT
        # supporting groupwise quantization where each tile has its own scaling factor
        if self.layer.weight_quant.is_groupwise:
            scales = pad_tensor_with_zeros(scales, self.max_accumulator_tile_size).view(
                -1, self.max_accumulator_tile_size)  # [Groups, OC * Tiles, IC / Tiles]
            scales = scales[:, 0]  # [Groups * OC * Tiles, 1]
            scales = scales.view(self.groups, -1,
                                 n_tiles).transpose(1, 2)  # [Groups, Tiles, OC/Groups]
        # else each tile has the same scaling factor (per-tensor or per-channel)
        else:
            scales = scales.view(self.groups, 1, -1)  # [Groups, 1, OC/Groups]
            scales = scales.repeat(1, n_tiles, 1)  # [Groups, Tiles, OC/Groups]
        thresholds *= scales  # translating centers back to the float range
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

        # List with permutation tensors for the Hessian and weight matrix.
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
                diag = torch.arange(self.columns, device='cpu')
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :], upper=True)
            h_inv = self.H
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H, self.B

        # initialize cumulative l1-norm
        a = torch.zeros_like(thresholds, device=dev)  # positive limits
        b = torch.zeros_like(thresholds, device=dev)  # negative limits

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, permutation_list[-1][i1:i2]],
                dtype=torch.float32)  # [groups, OC/groups, i2-i1]

            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                # need to apply soft thresholding and clamping before quantization
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    bx = perm[i1:i2][i] // self.max_accumulator_tile_size  # block index
                    # calculate the q_max and q_min for the right group and right block
                    # TODO: currently assuming round-to-zero; need to handle other rounding functions
                    q_max = scales[group_index, bx, :] * torch.clamp_min(
                        self.upper_lim - a[group_index, bx, :] - 0.5, 0.0)  # [OC/groups]
                    q_min = scales[group_index, bx, :] * torch.clamp_max(
                        self.lower_lim - b[group_index, bx, :] + 0.5, 0.0)  # [OC/groups]
                    q_arg = weight[group_index, :, perm[i1:i2][i]]  # [OC/groups]
                    # soft thresholding then clamping
                    q_arg = q_arg.sign() * torch.relu(
                        q_arg.abs() - thresholds[group_index, bx])  # [OC/groups]
                    q_arg.clamp_(q_min, q_max)  # clamping to bounds
                    weight[group_index, :, perm[i1:i2][i]] = q_arg
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [Groups, OC/groups]
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    q = q_groups[group_index].to(torch.float32)  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(
                            h_inv_block[group_index, i, i:].unsqueeze(0).to(dev))).to(dtype)
                # update the tracking mechanisms
                # TODO: need to handle non-zero zero points
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    bx = perm[i1:i2][i] // self.max_accumulator_tile_size  # block index
                    q = q_groups[group_index] / scales[group_index, bx]  # [OC/groups]
                    # increment cumulative l1-norm
                    a[group_index, bx, q >= 0] += q[q >= 0]
                    b[group_index, bx, q <= 0] += q[q <= 0]
                    assert (a <= self.upper_lim).all() and (a >= 0).all()
                    assert (b >= self.lower_lim).all() and (b <= 0).all()

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, "offload_params"):
            self.layer.offload_params(self.layer)

        del thresholds, scales  # memory management


class A2GPFQ(GPFQv2):
    """
    Memory-efficient, accumulator-aware GPFQ as proposed in https://arxiv.org/pdf/2409.17092
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            p,
            max_accumulator_bit_width,
            max_accumulator_tile_size) -> None:
        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig, p)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        if self.max_accumulator_tile_size is None:
            self.max_accumulator_tile_size = self.columns
        assert self.max_accumulator_tile_size > 2, "Error: accumulator tile size needs to be bigger than 2."
        assert self.max_accumulator_bit_width > 2, "Error: accumulator bit width needs to be bigger than 2."

    def single_layer_update(self, percdamp=0.01):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require quantized inputs."
        if self.quant_metadata is None:
            raise ValueError(
                "Expected self.quant_metadata to calculate accumualtor bounds, but recevied None. "
                "Make sure that either the input to the model is an IntQuantTensor or the layer has an input quant enabled. "
                "Also, check if `use_quant_activations=True` in `gpfq_mode` when `max_accumulator_bit_width` is specified. "
            )
        if hasattr(self.layer, "allocate_params"):
            self.layer.allocate_params(self.layer)
        weight: Tensor = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # TODO: add support for signed input activations
        if self.quant_metadata.signed:
            raise NotImplementedError("Signed inputs not yet supported.")

        n_tiles = math.ceil(weight.shape[-1] / self.max_accumulator_tile_size)
        scales: Tensor = self.layer.weight_quant.scale()
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                scales = scales.transpose(1, 0)  # This performs a view
            scales = scales.flatten(1)
        P = torch.tensor(self.max_accumulator_bit_width)
        N = self.quant_metadata.bit_width
        # NOTE: using sign-magnitude here, which is sufficient to support both
        # sign-magnitude and 2s complement accumulators
        self.upper_lim = (pow(2, P - 1) - 1) / float(pow(2, N) - 1)  # A
        self.lower_lim = -self.upper_lim  # B
        Z = (pow(2, P) - 2) / float(pow(2, N) - 1)  # l1-norm lim for zero-centered weight vector
        # translating into the quantized range; need to pad to get these thresholds
        wT = pad_tensor_with_zeros(weight / scales, self.max_accumulator_tile_size).view(
            -1, self.max_accumulator_tile_size)  # [OC * Tiles, IC / Tiles]
        thresholds = calc_average_nonzero_mag(
            wT - wT.mean(axis=1, keepdim=True), Z)  # [Groups * OC * Tiles]
        thresholds = thresholds.view(self.groups, -1,
                                     n_tiles).transpose(1, 2)  # [Groups, Tiles, OC/Groups]
        del wT
        # supporting groupwise quantization where each tile has its own scaling factor
        if self.layer.weight_quant.is_groupwise:
            scales = pad_tensor_with_zeros(scales, self.max_accumulator_tile_size).view(
                -1, self.max_accumulator_tile_size)  # [Groups, OC * Tiles, IC / Tiles]
            scales = scales[:, 0]  # [Groups * OC * Tiles, 1]
            scales = scales.view(self.groups, -1,
                                 n_tiles).transpose(1, 2)  # [Groups, Tiles, OC/Groups]
        # else each tile has the same scaling factor (per-tensor or per-channel)
        else:
            scales = scales.view(self.groups, 1, -1)  # [Groups, 1, OC/Groups]
            scales = scales.repeat(1, n_tiles, 1)  # [Groups, Tiles, OC/Groups]
        thresholds *= scales  # translating centers back to the float range

        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

        # initialize cumulative l1-norm
        a = torch.zeros_like(thresholds, device=dev)  # positive limit
        b = torch.zeros_like(thresholds, device=dev)  # negative limit

        # Try/Except in case the square root of H cannot be computed
        try:
            norms = torch.zeros((self.groups, self.columns), device=dev, dtype=torch.float32)
            self.H = self.H.to(dev)
            diag = torch.arange(self.columns, device='cpu')
            for i in range(self.groups):
                # stablize H with a dampening factor and then square root the matrix
                damp = percdamp * self.H[i].diag().mean()
                self.H[i, diag, diag] += damp
                norms[i] = self.H[i].diag()  # set the norms post-dampening
                eigvals, eigvecs = torch.linalg.eigh(self.H[i])
                eigvals.clamp_min_(0.0).sqrt_()  # should be positive-definite
                self.H[i] = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
            del eigvecs, eigvals, diag
            self.quant_input = self.H  # NOTE: do this here for the `get_permutation_list` function
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the matrix square root of H for layer {self.name} '
                f'GPFQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return

        # Try/Except in case the inverse of H cannot be computed
        try:
            self.float_input = self.H.clone()  # going to calculate H^{-1} here
            for i in range(self.groups):
                # from our matrix sqrt, we know G is symmetric and positive-definite, so we
                # can use Cholesky decomposition as an efficient, numerically stable inverse
                L = torch.linalg.cholesky(self.float_input[i])
                self.float_input[i] = torch.cholesky_inverse(L)
            self.float_input = torch.bmm(self.float_input.to(dev), self.G.to(dev))
            del L  # memory management
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of H for layer {self.name} '
                f'GPFQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H, self.G, self.B  # memory management

        permutation_list = self._get_permutation_list(weight)

        U = torch.zeros(
            weight.shape[0],
            weight.shape[1],
            self.float_input.shape[1],
            device=dev,
            dtype=torch.float32)  # [Groups, OC/groups, Samples]

        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                i = permutation_list[group_index][t]
                U[group_index] += torch.matmul(
                    weight[group_index, :, i].unsqueeze(1).to(torch.float32),
                    self.float_input[group_index, :, i].unsqueeze(0))
                norm = norms[group_index, i]
                if norm > 0:
                    q_arg = U[group_index].matmul(self.quant_input[group_index, :, i]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])
                bx = i // self.max_accumulator_tile_size  # block index
                q_arg = q_arg.sign() * torch.relu(
                    q_arg.abs() - thresholds[group_index, bx, :])  # soft thresholding

                # TODO: assuming round to nearest; need to generally support other rounding
                q_max = scales[group_index, bx] * torch.clamp_min(
                    self.upper_lim - a[group_index, bx, :] - 0.5, 0.0)
                q_min = scales[group_index, bx] * torch.clamp_max(
                    self.lower_lim - b[group_index, bx, :] + 0.5, 0.0)
                q_arg.clamp_(q_min, q_max)
                weight[group_index, :, i] = q_arg.to(dtype)
            q_groups: Tensor = self.get_quant_weights(t, 0, permutation_list)
            for group_index in range(self.groups):
                i = permutation_list[group_index][t]
                U[group_index] -= torch.matmul(
                    q_groups[group_index].unsqueeze(1).to(torch.float32),
                    self.quant_input[group_index, :, i].unsqueeze(0))
                bx = i // self.max_accumulator_tile_size  # block index
                q = q_groups[group_index] / scales[group_index, bx]  # [OC/groups]
                # increment cumulative l1-norm
                a[group_index, bx, q >= 0] += q[q >= 0]
                b[group_index, bx, q <= 0] += q[q <= 0]
                assert (a <= self.upper_lim).all() and (a >= 0).all()
                assert (b >= self.lower_lim).all() and (b <= 0).all()

        del self.quant_input, self.float_input
