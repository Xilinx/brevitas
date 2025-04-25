# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.utils.quant_utils import _CachedIO


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


class _AXE:

    quant_metadata: _CachedIO = None
    max_accumulator_bit_width: Tensor = None
    max_accumulator_tile_size: int = None

    @property
    def input_min(self):
        assert self.quant_metadata is not None, "Error: need quantized activations"
        input_bit_width = self.quant_metadata.bit_width
        input_is_signed = self.quant_metadata.signed
        # NOTE: can't get this from cache, so assuming worst-case scenario
        input_is_narrow = False
        input_min = min_int(input_is_signed, input_is_narrow, input_bit_width)
        assert input_min <= 0, f"Error: input_min={input_min}. Should be non-positive."
        return int(input_min)

    @property
    def input_max(self):
        assert self.quant_metadata is not None, "Error: need quantized activations"
        input_bit_width = self.quant_metadata.bit_width
        input_is_signed = self.quant_metadata.signed
        # NOTE: can't get this from cache, so assuming worst-case scenario
        input_is_narrow = False
        input_max = max_int(input_is_signed, input_is_narrow, input_bit_width)
        assert input_max >= 0, f"Error: input_max={input_max}. Should be non-negative."
        return int(input_max)

    def upper_lim(self, n: Tensor, p: Tensor):
        p0 = torch.exp2(self.max_accumulator_bit_width - 1.) - 1.
        p1 = (self.input_max * p) + (self.input_min * n)
        p2 = (p0 - p1) / self.input_max
        assert (p2 >= 0).all()

        # for unsigned data types, assuming round-to-nearest
        if self.input_min == 0:
            return p2 - 0.5

        n0 = -torch.exp2(self.max_accumulator_bit_width - 1.) + 1.
        n1 = (self.input_min * p) + (self.input_max * n)
        n2 = (n0 - n1) / self.input_min
        assert (n2 >= 0).all()

        # take the most restrictive lower limit (i.e., the smallest one),
        # note that we are assuming round-to-nearest here
        return torch.where(p2 < n2, p2, n2) - 0.5

    def lower_lim(self, n: Tensor, p: Tensor):
        n0 = -torch.exp2(self.max_accumulator_bit_width - 1.) + 1.
        n1 = (self.input_min * p) + (self.input_max * n)
        n2 = (n0 - n1) / self.input_max
        assert (n2 <= 0).all()

        # for unsigned data types, assuming round-to-nearest
        if self.input_min == 0:
            return n2 + 0.5

        p0 = torch.exp2(self.max_accumulator_bit_width - 1.) - 1.
        p1 = (self.input_max * p) + (self.input_min * n)
        p2 = (p0 - p1) / self.input_min
        assert (p2 <= 0).all()

        # take the most restrictive lower limit (i.e., the largest one),
        # note that we are assuming round-to-nearest here
        return torch.where(p2 > n2, p2, n2) + 0.5

    def get_scales_and_thresholds(self, weight: Tensor):
        # NOTE: assuming sign-magnitude here, which is sufficient to support both
        # sign-magnitude and 2s complement accumulators
        Z = (torch.exp2(self.max_accumulator_bit_width) -
             2) / float(self.input_max - self.input_min)
        n_tiles = math.ceil(weight.shape[-1] / self.max_accumulator_tile_size)

        scales: Tensor = self.layer.weight_quant.scale()
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                scales = scales.transpose(1, 0)  # This performs a view
            scales = scales.flatten(1)

        # translating into the quantized range; need to pad to get these thresholds
        wT = pad_tensor_with_zeros(weight / scales, self.max_accumulator_tile_size).view(
            -1, self.max_accumulator_tile_size)  # [OC * Tiles, IC / Tiles]
        # calculate the thresholds after zero centering projection
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
        return thresholds, scales


class A2GPTQ(_AXE, GPTQ):
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
        assert max_accumulator_bit_width is not None, \
            "Error: max_accumulator_bit_width is not specified."
        if not isinstance(max_accumulator_bit_width, Tensor):
            max_accumulator_bit_width = torch.tensor(max_accumulator_bit_width)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        if self.max_accumulator_tile_size is None:
            self.max_accumulator_tile_size = self.columns
        assert self.max_accumulator_tile_size > 2, \
            "Error: accumulator tile size needs to be bigger than 2."
        assert self.max_accumulator_bit_width > 2, \
            "Error: accumulator bit width needs to be bigger than 2."

    def single_layer_update(self, percdamp=0.01, c=1e4):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPTQ does not support weight quantizers that require quantized inputs."
        if self.quant_metadata is None:
            raise ValueError(
                "Expected self.quant_metadata to calculate accumulator bounds, but recevied None. "
                "Make sure that either the input to the model is an IntQuantTensor or the layer has an input quant enabled. "
                "Also, check if `use_quant_activations=True` in `gptq_mode` when `max_accumulator_bit_width` is specified. "
            )
        if hasattr(self.layer, "allocate_params"):
            self.layer.allocate_params(self.layer)
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the
        # original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)

        # TODO: currently assuming round-to-nearest; need to handle other
        # rounding functions
        rounding_mode = self.layer.weight_quant.rounding_mode
        if rounding_mode.lower() != "round":
            raise NotImplementedError(f"{rounding_mode} not yet supported.")

        thresholds, scales = self.get_scales_and_thresholds(weight)
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
                # stabilizing the Cholesky decomposition with a fairly large constant, c
                self.H[i, :, :] = torch.linalg.cholesky(
                    self.H[i, :, :] * c, upper=True) / math.sqrt(c)
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
        lim_dtype = torch.int32 if self.max_accumulator_bit_width < 33 else torch.int64
        pos_limits = torch.zeros_like(thresholds, device=dev, dtype=lim_dtype)  # positive limits
        neg_limits = torch.zeros_like(thresholds, device=dev, dtype=lim_dtype)  # negative limits
        max_limits = ((2 ** (self.max_accumulator_bit_width.to(lim_dtype) - 1)) - 1)

        for i1 in tqdm(range(0, self.columns, self.blocksize), desc="Neurons: ", leave=False):
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
                    s = scales[group_index, bx].to(dtype)
                    n = neg_limits[group_index, bx]
                    p = pos_limits[group_index, bx]
                    q_arg: Tensor = weight[group_index, :,
                                           perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
                    u = self.upper_lim(n, p)
                    l = self.lower_lim(n, p)
                    assert (u - l + 1 >= 0).all()
                    q_max = s * torch.clamp_min(u, 0.0)  # [OC/groups]
                    q_min = s * torch.clamp_max(l, 0.0)  # [OC/groups]
                    # soft thresholding then clamping
                    q_arg = q_arg.sign() * torch.relu(
                        q_arg.abs() - thresholds[group_index, bx])  # [OC/groups]
                    q_arg.clamp_(q_min, q_max)  # clamping to bounds
                    weight[group_index, :, perm[i1:i2][i]] = q_arg.to(dtype)
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
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    bx = perm[i1:i2][i] // self.max_accumulator_tile_size  # block index
                    q = q_groups[group_index] / scales[group_index, bx]  # [OC/groups]
                    # increment cumulative l1-norm
                    pos_limits[group_index, bx, q >= 0] += q[q >= 0].to(lim_dtype)
                    neg_limits[group_index, bx, q <= 0] += q[q <= 0].to(lim_dtype)
                    assert (pos_limits >= 0).all()
                    assert (neg_limits <= 0).all()
                    assert (((self.input_max * pos_limits) +
                             (self.input_min * neg_limits)) <= max_limits).all()
                    assert (
                        -((self.input_min * pos_limits) +
                          (self.input_max * neg_limits)) <= max_limits).all()

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, "offload_params"):
            self.layer.offload_params(self.layer)

        del thresholds, scales  # memory management


class A2GPFQ(_AXE, GPFQ):
    """
    Optimized version of accumulator-aware GPFQ as proposed in https://arxiv.org/pdf/2409.17092
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            max_accumulator_bit_width,
            max_accumulator_tile_size) -> None:
        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig)
        assert max_accumulator_bit_width is not None, \
            "Error: max_accumulator_bit_width must be specified."
        if not isinstance(max_accumulator_bit_width, Tensor):
            max_accumulator_bit_width = torch.tensor(max_accumulator_bit_width)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        if self.max_accumulator_tile_size is None:
            self.max_accumulator_tile_size = self.columns
        assert self.max_accumulator_tile_size > 2, \
            "Error: accumulator tile size needs to be bigger than 2."
        assert self.max_accumulator_bit_width > 2, \
            "Error: accumulator bit width needs to be bigger than 2."

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require quantized inputs."
        if self.quant_metadata is None:
            raise ValueError(
                "Expected self.quant_metadata to calculate accumulator bounds, but recevied None. "
                "Make sure that either the input to the model is an IntQuantTensor or the layer has an input quant enabled. "
                "Also, check if `use_quant_activations=True` in `gpfq_mode` when `max_accumulator_bit_width` is specified. "
            )
        if hasattr(self.layer, "allocate_params"):
            self.layer.allocate_params(self.layer)
        del self.B  # memory management

        weight = self.layer.weight.data
        if self.create_weight_orig:
            weight_orig = self.layer.weight_orig.data
        else:
            warnings.warn("Warning: GPFQ will perform better with `create_weight_orig=True`.")
            weight_orig = weight.clone()
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

        # TODO: currently assuming round-to-nearest; need to handle other
        # rounding functions
        rounding_mode = self.layer.weight_quant.rounding_mode
        if rounding_mode.lower() != "round":
            raise NotImplementedError(f"{rounding_mode} not yet supported.")

        thresholds, scales = self.get_scales_and_thresholds(weight)
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

        Dg: Tensor = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        Dh: Tensor = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        for group_index in range(self.groups):
            Dg[group_index].copy_(self.G[group_index].diag())
            Dh[group_index].copy_(self.H[group_index].diag())
        # if either norms are 0, the weight is effectively pruned
        Ds = torch.where(Dg * Dh != 0, Dg / Dh, torch.zeros_like(Dg))  # \hat{D}_tt / D_tt

        Lg: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                 device=dev,
                                 dtype=torch.float32)
        Lh: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                 device=dev,
                                 dtype=torch.float32)
        for group_index in range(self.groups):
            L0g = torch.tril(self.G[group_index], -1)  # L0
            L0h = torch.tril(self.H[group_index], -1)  # \hat{L0}
            Dhi = torch.where(
                Dh[group_index] != 0, 1. / Dh[group_index],
                torch.zeros_like(Dh[group_index]))  # D^{-1}
            Lg[group_index].copy_(torch.diag(Dhi) @ L0g)
            Lh[group_index].copy_(torch.diag(Dhi) @ L0h)

        del self.H, self.G  # memory management

        # initialize cumulative l1-norm
        lim_dtype = torch.int32 if self.max_accumulator_bit_width < 33 else torch.int64
        pos_limits = torch.zeros_like(thresholds, device=dev, dtype=lim_dtype)  # positive limits
        neg_limits = torch.zeros_like(thresholds, device=dev, dtype=lim_dtype)  # negative limits
        max_limits = ((2 ** (self.max_accumulator_bit_width.to(lim_dtype) - 1)) - 1)

        for t in range(weight.shape[-1]):
            q_groups = self.get_quant_weights(t, 0, permutation_list, with_quant_history=True)
            for group_index in range(self.groups):
                # t := time step (Lg, Lh, and Ds are re-ordered in time)
                # i := input channel index (weight and error are not re-ordered)
                # bx := block index (for accumulation)
                perm = permutation_list[group_index]
                i = perm[t]
                bx = i // self.max_accumulator_tile_size
                w = weight_orig[group_index, :, perm[:t]].to(torch.float32)
                q = q_groups[group_index].to(torch.float32)
                Lw = w.matmul(Lg[group_index, t, :t])
                Lq = q.matmul(Lh[group_index, t, :t])
                q_arg = Ds[group_index, t] * weight[group_index, :, i].to(torch.float32) + Lw - Lq
                assert not torch.isnan(q_arg).any()

                # calculate the q_max and q_min for the right group and right block
                s = scales[group_index, bx].to(torch.float32)
                n = neg_limits[group_index, bx]
                p = pos_limits[group_index, bx]
                u = self.upper_lim(n, p)
                l = self.lower_lim(n, p)
                assert (u - l + 1 >= 0).all()
                q_max = s * torch.clamp_min(u, 0.0)  # [OC/groups]
                q_min = s * torch.clamp_max(l, 0.0)  # [OC/groups]
                # soft thresholding then clamping
                q_arg = q_arg.sign() * torch.relu(
                    q_arg.abs() - thresholds[group_index, bx])  # [OC/groups]
                q_arg.clamp_(q_min, q_max)  # clamping to bounds

                weight[group_index, :, i] = q_arg.to(dtype)

            # update the tracking mechanisms
            q_groups = self.get_quant_weights(t, 0, permutation_list)  # [Groups, OC/groups]
            for group_index in range(self.groups):
                i = permutation_list[group_index][t]
                bx = i // self.max_accumulator_tile_size  # block index
                q = q_groups[group_index] / scales[group_index, bx]  # [OC/groups]
                # increment cumulative l1-norm
                pos_limits[group_index, bx, q >= 0] += q[q >= 0].to(lim_dtype)
                neg_limits[group_index, bx, q <= 0] += q[q <= 0].to(lim_dtype)
                assert (pos_limits >= 0).all()
                assert (neg_limits <= 0).all()
                assert (((self.input_max * pos_limits) +
                         (self.input_min * neg_limits)) <= max_limits).all()
                assert (
                    -((self.input_min * pos_limits) +
                      (self.input_max * neg_limits)) <= max_limits).all()

        del thresholds, scales
