# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import List, Optional
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import unfoldNd

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
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
            if is_conv_transposed(self.layer):
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


class gpfq_mode(gpxq_mode):
    """
    Apply GPFQ algorithm.

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
            gpfq_class: GPFQ = GPFQ) -> None:
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

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Disable quantization
        # TODO: Ensure that removing is_training=False does not cause any regression and remove,
        # if that is the case
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

        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            out = self.orig_forward(*args, **kwargs)
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = False
            return out

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        return self.gpfq_class(
            layer=layer,
            name=name,
            act_order=self.act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig)
