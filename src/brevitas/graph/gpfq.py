# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import Callable, List, Optional
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import unfoldNd

from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.graph.gpxq import AXE
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.gpxq import SUPPORTED_TCONV_OP
import brevitas.nn as qnn
from brevitas.utils.torch_utils import StopFwdException


class GPFQ(GPxQ):
    """
    Optimized greedy path following quantization (GPFQ)

    See `Post-training Quantization for Neural Networks with Provable Guarantees`

    https://epubs.siam.org/doi/abs/10.1137/22M1511709
    """

    def __init__(self, layer, name, act_order, len_parallel_layers, create_weight_orig) -> None:
        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig)
        # Initialize covariance matrices. We need them in float32
        # H = \hat{X} \hat{X}^T
        self.H: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32)
        # G = \hat{X} X^T
        self.G: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32)
        # buffer to speed-up GPU to CPU transfer
        self.B: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32,
                                     pin_memory=torch.cuda.is_available())
        self.nsamples = 0

        self.quant_input = None

        self.create_weight_orig = create_weight_orig  # not saved by base class

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        is_quant_enabled = module.weight_quant.is_quant_enabled

        inp = self.process_input(input)

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            inp = inp.t()
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                unfold_impl = unfoldNd.UnfoldTransposeNd
            else:
                unfold_impl = unfoldNd.UnfoldNd

            unfold = unfold_impl(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for inp in inp_by_group:
                inp = unfold(inp)
                inp = inp.transpose(1, 0)
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        # Normalizing by the sequence length for numerical stability
        n = inp_processed.size(1)
        inp_processed = math.sqrt(1 / n) * inp_processed.to(torch.float32)

        # NOTE: in the gpfq_mode context manager, we first collect quant inputs, then
        # we collect float inputs for the same batch. We assume this pattern here, but
        # will add a check just in case.

        # if quant is not enabled, then it is the float input; if it is a float input
        # then a quant input has already happened and we can update G
        if not is_quant_enabled:
            # Computing the normalized G matrix using CPU buffer
            self.B.copy_(self.quant_input.bmm(inp_processed.transpose(2, 1)))
            self.G += self.B
            self.quant_input = None  # NOTE: set back to None now that we've used it
        else:
            # Computing the normalized H matrix using CPU buffer
            self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
            self.H += self.B
            # store the quantized input for computing the H matrix
            assert self.quant_input is None
            self.quant_input = inp_processed

        # If we are executing GPFQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require metadata from input quantizers."

        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        del self.B  # free up memory by deleting the buffer

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
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
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

        Dg = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        Dh = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        for group_index in range(self.groups):
            Dg[group_index].copy_(self.G[group_index].diag())
            Dh[group_index].copy_(self.H[group_index].diag())
        # if either norms are 0, the weight is effectively pruned
        Ds = torch.where(Dg * Dh != 0, Dg / Dh, torch.zeros_like(Dg))  # \hat{D}_tt / D_tt

        Lg = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=torch.float32)
        Lh = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=torch.float32)
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
                w = weight_orig[group_index, :, permutation_list[group_index][:t]].to(torch.float32)
                q = q_groups[group_index].to(torch.float32)
                Lw = w.matmul(Lg[group_index, t, :t])
                Lq = q.matmul(Lh[group_index, t, :t])
                q_arg = Ds[group_index, t] * weight[group_index, :, i].to(torch.float32) + Lw - Lq
                assert not torch.isnan(q_arg).any()
                weight[group_index, :, i] = q_arg.to(dtype)

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class A2GPFQ(AXE, GPFQ):
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
        assert self.max_accumulator_tile_size > 1, \
            "Error: accumulator tile size needs to be bigger than 1."
        assert self.max_accumulator_bit_width > 1, \
            "Error: accumulator bit width needs to be bigger than 1."

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
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
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

        Dg = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        Dh = torch.zeros((self.groups, self.columns), dtype=torch.float32)
        for group_index in range(self.groups):
            Dg[group_index].copy_(self.G[group_index].diag())
            Dh[group_index].copy_(self.H[group_index].diag())
        # if either norms are 0, the weight is effectively pruned
        Ds = torch.where(Dg * Dh != 0, Dg / Dh, torch.zeros_like(Dg))  # \hat{D}_tt / D_tt

        Lg = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=torch.float32)
        Lh = torch.zeros((self.groups, self.columns, self.columns), device=dev, dtype=torch.float32)
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
                pos_max_limit = ((self.input_max * pos_limits) + (self.input_min * neg_limits))
                assert (pos_max_limit <= max_limits).all(), \
                    f"pos_max_limit: {pos_max_limit.max()}, max_limits: {max_limits}"
                assert (
                    -((self.input_min * pos_limits) +
                      (self.input_max * neg_limits)) <= max_limits).all()

        del thresholds, scales


class gpfq_mode(gpxq_mode):
    """
    Apply GPFQ algorithm https://epubs.siam.org/doi/abs/10.1137/22M1511709
    Or accumulator-aware GPFQ (A2GPFQ) algorithm https://arxiv.org/pdf/2409.17092

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
        gpfq_class (GPFQ): The uninitialized class to perform GPFQ.
            Default: `brevitas.graph.gpfq.GPFQv2`, which is the memory-efficient formulation
        a2q_layer_filter_fnc (Callable[[Module], bool]): A function that takes a layer and returns True
            if the layer should be quantized with A2GPFQ. Default: lambda x: True
        max_accumulator_bit_width (Optional[int]): The maximum bit width of the accumulator. Default: None
        max_accumulator_tile_size (Optional[int]): The maximum tile size for accumulator-aware quantization.
            If `None` and `max_accumulator_bit_width` is specified, then a monolithic accumulator is
            assumed (see `Accumulator-Aware Post-Training Quantization` for more details). Default: None

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
            gpfq_class: GPFQ = GPFQ,
            a2q_layer_filter_fnc: Optional[Callable[[nn.Module], bool]] = lambda x: True,
            max_accumulator_bit_width: Optional[int] = None,
            max_accumulator_tile_size: Optional[int] = None) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(
            model,
            group_of_parallel_layers,
            inplace,
            create_weight_orig,
            use_quant_activations,
            act_order,
            return_forward_output)

        self.gpfq_class = gpfq_class
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Disable quantization
        self.return_quant_tensor_state = disable_return_quant_tensor(self.model)
        self.disable_quant_inference.disable_param_quantization(self.model, is_training=False)
        self.disable_quant_inference.disable_act_quantization(self.model, is_training=False)
        # Collect float input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Re-enable quantization. If activation quantization is disabled,
        # we also disable bias quantization
        self.disable_quant_inference.enable_param_quantization(self.model, is_training=False)
        if self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(self.model, is_training=False)
        else:
            self.disable_quant_inference.disable_bias_quantization(self.model, is_training=False)
        restore_return_quant_tensor(self.model, self.return_quant_tensor_state)

        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            out = self.orig_forward(*args, **kwargs)
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = False
            return out

    def requires_accumulator_awareness(self, layer):
        # if the accumulator bit width is specified and the layer determined by the filter
        # then quantize with A2GPFQ
        if (self.max_accumulator_bit_width is not None) and self.a2q_layer_filter_fnc(layer):
            return True
        return False

    def initialize_module_optimizer(
            self, layer, name, act_order, len_parallel_layers, create_weight_orig):
        if self.requires_accumulator_awareness(layer):
            return A2GPFQ(
                layer=layer,
                name=name,
                act_order=act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                max_accumulator_bit_width=self.max_accumulator_bit_width,
                max_accumulator_tile_size=self.max_accumulator_tile_size)
        return self.gpfq_class(
            layer=layer,
            name=name,
            act_order=act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig)
