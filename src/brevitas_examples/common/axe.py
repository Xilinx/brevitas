# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Union
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.utils.quant_utils import _CachedIO
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


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


class AXEMixin:
    """
    Accumulator-aware extensions for greedy path sequential quantization algorithms
    such as the GPxQ family of algorithms.

    See "Accumulator-Aware Post-Training Quantization for Large Language Models" for more details.
      https://openreview.net/forum?id=p6l0579yj7
    """

    quant_metadata: Union[_CachedIO, _CachedIOGroupwiseInt] = None

    def __init__(
            self,
            max_accumulator_bit_width: Union[int, Tensor],
            max_accumulator_tile_size: Optional[int] = None):

        if max_accumulator_bit_width is None:
            raise ValueError("max_accumulator_bit_width is not specified.")
        if not isinstance(max_accumulator_bit_width, Tensor):
            max_accumulator_bit_width = torch.tensor(max_accumulator_bit_width)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        if self.max_accumulator_bit_width <= 2:
            raise ValueError(
                f"accumulator bit width needs to be bigger than 2, received {self.max_accumulator_bit_width}"
            )

        self.max_accumulator_tile_size = max_accumulator_tile_size
        if self.max_accumulator_tile_size is None:
            self.max_accumulator_tile_size = self.columns
        if self.max_accumulator_tile_size <= 1:
            raise ValueError(
                f"accumulator tile size needs to be bigger than 1, received {self.max_accumulator_tile_size}"
            )
        if self.layer.weight_quant.is_groupwise:
            # AXE constrains the weights against a target accumulator for a full dot product, and
            # the math assumes one scale per dot product. A dot product can be split into tiles
            # (partial dot products) with one scale each, so with groupwise scales we require the
            # group size to equal the tile size (or a monolithic accumulator spanning the row).
            if (self.max_accumulator_tile_size != self.columns) \
                and (self.max_accumulator_tile_size != self.layer.weight_quant.group_size):
                raise ValueError(
                    "Error: only supporting accumulator-aware groupwise weight quantization"
                    "when the group size is equal to the accumulator tile size or a monolithic"
                    "accumulator is assumed (i.e., `max_accumulator_tile_size=None`).")
            # GPxQ (and AXE) unroll a convolution into a single dot product, but Brevitas' MX
            # implementation currently forms groups only along the input-channel dimension, which
            # no longer maps onto that unrolled dot product. We leave resolving this to the future.
            if isinstance(self.layer, SUPPORTED_CONV_OP):
                raise ValueError(
                    "Error: accumulator-aware quantization with groupwise weight scales is not "
                    "supported for convolutions.")

    def reshape_gpxq_weights(self, weight):
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        return weight

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

        if self.input_min == 0:
            return p2

        n0 = -torch.exp2(self.max_accumulator_bit_width - 1.) + 1.
        n1 = (self.input_min * p) + (self.input_max * n)
        n2 = (n0 - n1) / self.input_min
        assert (n2 >= 0).all()

        # take the most restrictive upper limit (i.e., the smallest one)
        return torch.where(p2 < n2, p2, n2)

    def lower_lim(self, n: Tensor, p: Tensor):
        n0 = -torch.exp2(self.max_accumulator_bit_width - 1.) + 1.
        n1 = (self.input_min * p) + (self.input_max * n)
        n2 = (n0 - n1) / self.input_max
        assert (n2 <= 0).all()

        if self.input_min == 0:
            return n2

        p0 = torch.exp2(self.max_accumulator_bit_width - 1.) - 1.
        p1 = (self.input_max * p) + (self.input_min * n)
        p2 = (p0 - p1) / self.input_min
        assert (p2 <= 0).all()

        # take the most restrictive lower limit (i.e., the largest one)
        return torch.where(p2 > n2, p2, n2)

    def get_thresholds(self, weight: Tensor, scales: Tensor, n_tiles: int) -> Tensor:
        """
        Per-tile soft-thresholding radius: the largest magnitude that can be shrunk to zero
        while keeping each accumulator tile within budget. Computed in the integer domain via
        an L1-ball projection, then mapped back to the float domain.

        weight, scales: [Groups, OC/Groups, IC]
        returns thresholds: [Groups, n_tiles, OC/Groups] in float domain
        """
        Z = (torch.exp2(self.max_accumulator_bit_width) - 2) / float(
            self.input_max - self.input_min)  # L1 radius (accumulator budget)
        w_int = (weight / scales).to(torch.float32)  # [Groups, OC/Groups, IC]
        wT = pad_tensor_with_zeros(
            w_int.flatten(0, 1),  # [Groups*OC/Groups, IC]
            self.max_accumulator_tile_size,
        ).view(-1, self.max_accumulator_tile_size)  # [Groups*OC/Groups*n_tiles, tile_size]
        thresholds = calc_average_nonzero_mag(
            wT - wT.mean(dim=1, keepdim=True), Z)  # [Groups*OC/Groups*n_tiles]
        thresholds = thresholds.view(self.groups, -1,
                                     n_tiles).transpose(1, 2)  # [Groups, n_tiles, OC/Groups]
        # scale back to float domain: one scale per (group, tile, OC) — take first IC in each tile
        s_per_tile = scales[:, :, ::self.max_accumulator_tile_size]  # [Groups, OC/Groups, n_tiles]
        thresholds *= s_per_tile.permute(0, 2, 1)  # [Groups, n_tiles, OC/Groups]
        return thresholds


class axe_mode_mixin:
    """
    Mixin to inject accumulator-aware dispatch into a gpxq_mode-derived context manager.
    """

    max_accumulator_bit_width: Optional[int] = None
    max_accumulator_tile_size: Optional[int] = None
    a2q_layer_filter_fnc = staticmethod(lambda x: True)

    def is_valid_a2q_layer(self, layer) -> bool:
        """Check if a layer is valid for accumulator-aware quantization (A2Q)"""
        # We don't apply A2Q if the bit width is not specified (default is None)
        if self.max_accumulator_bit_width is None:
            return False
        # We expose a filter function to enable/disable A2Q based on layer characteristics
        if not self.a2q_layer_filter_fnc(layer):
            return False
        return True


class A2GPTQ(AXEMixin, GPTQ):
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
            max_accumulator_tile_size,
            device='cpu',
            dtype=torch.float32) -> None:
        GPTQ.__init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            num_blocks,
            device=device,
            dtype=dtype)
        AXEMixin.__init__(self, max_accumulator_bit_width, max_accumulator_tile_size)

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
        if self.use_intermediate_buffer:
            del self.B  # free memory
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the
        # original dtype
        dtype = weight.dtype

        scales = self.layer.quant_weight().scale
        scales = scales.broadcast_to(weight.shape)
        if scales.ndim > 0:
            scales = self.reshape_gpxq_weights(scales)  # [OC, IC]
        weight = self.reshape_gpxq_weights(weight)  # [OC, IC]

        scales = scales.view(self.groups, -1, weight.shape[-1])
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

        # List with permutation tensors for the Hessian and weight matrix.
        # If act_order is False, the tensors will be ordered indexes.
        # For groupwise convolution, we have one tensor per group,
        # thus len(permutation_list) is always equal to self.groups.
        # We do not explicity permute the weight matrix, only the Hessian.
        permutation_list = []
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
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

        n_tiles = math.ceil(weight.shape[-1] / self.max_accumulator_tile_size)
        get_block_index = lambda bx: bx // self.max_accumulator_tile_size

        thresholds = self.get_thresholds(weight, scales, n_tiles)  # [Groups, n_tiles, OC/Groups]

        # initialize cumulative l1-norm
        lim_dtype = torch.int32 if self.max_accumulator_bit_width < 33 else torch.int64
        pos_limits = torch.zeros((self.groups, n_tiles, weight.shape[1]),
                                 device=dev,
                                 dtype=lim_dtype)  # positive limits
        neg_limits = torch.zeros((self.groups, n_tiles, weight.shape[1]),
                                 device=dev,
                                 dtype=lim_dtype)  # negative limits
        max_limits = ((2 ** (self.max_accumulator_bit_width.to(lim_dtype) - 1)) - 1)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, permutation_list[-1][i1:i2]],
                dtype=self.dtype)  # [groups, OC/groups, i2-i1]
            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                # need to apply soft thresholding and clamping before quantization
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    block_index = get_block_index(perm[i1:i2][i])  # block index
                    # calculate the q_max and q_min for the right group and right block
                    n = neg_limits[group_index, block_index]
                    p = pos_limits[group_index, block_index]
                    s = scales[group_index, :, perm[i1:i2][i]].to(self.dtype)
                    q_arg = weight[group_index, :, perm[i1:i2][i]].to(self.dtype)  # [OC/groups]
                    u = self.upper_lim(n, p)
                    l = self.lower_lim(n, p)
                    assert (u - l + 1 >= 0).all()
                    q_max = s * torch.clamp_min(u, 0.0).floor()  # [OC/groups]
                    q_min = s * torch.clamp_max(l, 0.0).ceil()  # [OC/groups]
                    # soft thresholding then clamping
                    q_arg = q_arg.sign() * torch.relu(
                        q_arg.abs() - thresholds[group_index, block_index])
                    q_arg.clamp_(q_min, q_max)  # clamping to bounds
                    weight[group_index, :, perm[i1:i2][i]] = q_arg.to(dtype)
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [Groups, OC/groups]
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
                # update the tracking mechanisms
                for group_index in range(self.groups):
                    perm = permutation_list[group_index]
                    block_index = get_block_index(perm[i1:i2][i])
                    s = scales[group_index, :, perm[i1:i2][i]].to(self.dtype)
                    # round before the integer cast: q_groups is the dequantized weight in
                    # the model dtype (e.g. bf16), so q_groups / s is not exactly integer and
                    # a direct int cast would truncate, undercounting the accumulator l1-norm
                    q = (q_groups[group_index].to(self.dtype) / s).round()  # [OC/groups]
                    # increment cumulative l1-norm
                    pos_limits[group_index, block_index, q >= 0] += q[q >= 0].to(lim_dtype)
                    neg_limits[group_index, block_index, q <= 0] += q[q <= 0].to(lim_dtype)

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, "offload_params"):
            self.layer.offload_params(self.layer)


class A2GPFQ(AXEMixin, GPFQ):
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
            max_accumulator_tile_size,
            device='cpu',
            dtype=torch.float32) -> None:
        GPFQ.__init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            device=device,
            dtype=dtype)
        AXEMixin.__init__(self, max_accumulator_bit_width, max_accumulator_tile_size)

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require quantized inputs."
        assert hasattr(self.layer, 'weight_orig'), \
            "Error: GPFQ requires the original weights to be stored, see `create_weight_orig`."
        if self.quant_metadata is None:
            raise ValueError(
                "Expected self.quant_metadata to calculate accumulator bounds, but recevied None. "
                "Make sure that either the input to the model is an IntQuantTensor or the layer has an input quant enabled. "
                "Also, check if `use_quant_activations=True` in `gpfq_mode` when `max_accumulator_bit_width` is specified. "
            )
        if hasattr(self.layer, "allocate_params"):
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

        scales = self.layer.quant_weight().scale
        scales = scales.broadcast_to(weight.shape)
        if scales.ndim > 0:
            scales = self.reshape_gpxq_weights(scales)  # [OC, IC]
        weight = self.reshape_gpxq_weights(weight)  # [OC, IC]
        weight_orig = self.reshape_gpxq_weights(weight_orig)

        weight_orig = weight_orig.view(self.groups, -1, weight.shape[-1])
        scales = scales.view(self.groups, -1, weight.shape[-1])
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

        Dg: Tensor = torch.zeros((self.groups, self.columns), dtype=self.dtype, device=self.device)
        Dh: Tensor = torch.zeros((self.groups, self.columns), dtype=self.dtype, device=self.device)
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

        n_tiles = math.ceil(weight.shape[-1] / self.max_accumulator_tile_size)
        get_block_index = lambda bx: bx // self.max_accumulator_tile_size

        thresholds = self.get_thresholds(weight, scales, n_tiles)  # [Groups, n_tiles, OC/Groups]

        # initialize cumulative l1-norm
        lim_dtype = torch.int32 if self.max_accumulator_bit_width < 33 else torch.int64
        pos_limits = torch.zeros((self.groups, n_tiles, weight.shape[1]),
                                 device=dev,
                                 dtype=lim_dtype)  # positive limits
        neg_limits = torch.zeros((self.groups, n_tiles, weight.shape[1]),
                                 device=dev,
                                 dtype=lim_dtype)  # negative limits
        max_limits = ((2 ** (self.max_accumulator_bit_width.to(lim_dtype) - 1)) - 1)

        for t in range(weight.shape[-1]):
            q_groups = self.get_quant_weights(t, 0, permutation_list, with_quant_history=True)
            for group_index in range(self.groups):
                # t := time step (Lg, Lh, and Ds are re-ordered in time)
                # i := input channel index (weight and error are not re-ordered)
                # block_index := block index for accumulation
                perm = permutation_list[group_index]
                i = perm[t]
                block_index = get_block_index(i)
                w = weight_orig[group_index, :, perm[:t]].to(self.dtype)
                q = q_groups[group_index].to(self.dtype)
                Lw = w.matmul(Lg[group_index, t, :t])
                Lq = q.matmul(Lh[group_index, t, :t])
                q_arg = Ds[group_index, t] * weight[group_index, :, i].to(self.dtype) + Lw - Lq
                assert not torch.isnan(q_arg).any()

                # calculate the q_max and q_min for the right group and right block
                s = scales[group_index, :, i].to(self.dtype)
                n = neg_limits[group_index, block_index]
                p = pos_limits[group_index, block_index]
                u = self.upper_lim(n, p)
                l = self.lower_lim(n, p)
                assert (u - l + 1 >= 0).all()
                q_max = s * torch.clamp_min(u, 0.0).floor()  # [OC/groups]
                q_min = s * torch.clamp_max(l, 0.0).ceil()  # [OC/groups]
                # soft thresholding then clamping
                q_arg = q_arg.sign() * torch.relu(
                    q_arg.abs() - thresholds[group_index, block_index])
                q_arg.clamp_(q_min, q_max)  # clamping to bounds

                weight[group_index, :, i] = q_arg.to(dtype)

            # update the tracking mechanisms
            q_groups = self.get_quant_weights(t, 0, permutation_list)  # [Groups, OC/groups]
            for group_index in range(self.groups):
                i = permutation_list[group_index][t]
                block_index = get_block_index(i)  # block index
                s = scales[group_index, :, i].to(self.dtype)
                # round before the integer cast: q_groups is the dequantized weight in
                # the model dtype (e.g. bf16), so q_groups / s is not exactly integer and
                # a direct int cast would truncate, undercounting the accumulator l1-norm
                q = (q_groups[group_index].to(self.dtype) / s).round()  # [OC/groups]
                # increment cumulative l1-norm
                pos_limits[group_index, block_index, q >= 0] += q[q >= 0].to(lim_dtype)
                neg_limits[group_index, block_index, q <= 0] += q[q <= 0].to(lim_dtype)

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class a2gptq_mode(axe_mode_mixin, gptq_mode):
    """
    Accumulator-aware variant of `gptq_mode`. Dispatches to `A2GPTQ` for layers
    that pass `a2q_layer_filter_fnc`; falls back to the configured `gptq_class`
    otherwise.
    """

    def __init__(
            self,
            model: Module,
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
            a2q_layer_filter_fnc: Optional[Callable[[Module], bool]] = lambda x: True,
            max_accumulator_bit_width: Optional[int] = None,
            max_accumulator_tile_size: Optional[int] = None) -> None:
        gptq_mode.__init__(
            self,
            model=model,
            group_of_parallel_layers=group_of_parallel_layers,
            inplace=inplace,
            create_weight_orig=create_weight_orig,
            use_quant_activations=use_quant_activations,
            num_blocks=num_blocks,
            return_forward_output=return_forward_output,
            act_order=act_order,
            gptq_class=gptq_class,
            device=device,
            dtype=dtype)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        if self.is_valid_a2q_layer(layer):
            return A2GPTQ(
                layer=layer,
                name=name,
                act_order=self.act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                num_blocks=self.num_blocks,
                max_accumulator_bit_width=self.max_accumulator_bit_width,
                max_accumulator_tile_size=self.max_accumulator_tile_size,
                device=self.device,
                dtype=self.dtype)
        return super().initialize_module_optimizer(
            layer, name, len_parallel_layers, create_weight_orig)


class a2gpfq_mode(axe_mode_mixin, gpfq_mode):
    """
    Accumulator-aware variant of `gpfq_mode`. Dispatches to `A2GPFQ` for layers
    that pass `a2q_layer_filter_fnc`; falls back to the configured `algorithm_impl`
    otherwise.
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
            a2q_layer_filter_fnc: Optional[Callable[[nn.Module], bool]] = lambda x: True,
            max_accumulator_bit_width: Optional[int] = None,
            max_accumulator_tile_size: Optional[int] = None) -> None:
        gpfq_mode.__init__(
            self,
            model=model,
            group_of_parallel_layers=group_of_parallel_layers,
            inplace=inplace,
            create_weight_orig=create_weight_orig,
            use_quant_activations=use_quant_activations,
            return_forward_output=return_forward_output,
            act_order=act_order,
            algorithm_impl=algorithm_impl,
            device=device,
            dtype=dtype)
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        if self.is_valid_a2q_layer(layer):
            return A2GPFQ(
                layer=layer,
                name=name,
                act_order=self.act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                max_accumulator_bit_width=self.max_accumulator_bit_width,
                max_accumulator_tile_size=self.max_accumulator_tile_size,
                device=self.device,
                dtype=self.dtype)
        return super().initialize_module_optimizer(
            layer, name, len_parallel_layers, create_weight_orig)
