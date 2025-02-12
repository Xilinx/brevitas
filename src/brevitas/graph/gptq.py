# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import Callable, List, Optional
import warnings

from packaging import version
import torch
from torch import Tensor
from torch.nn import Module

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError
import unfoldNd

from brevitas import torch_version
from brevitas.graph.gpxq import AXE
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.gpxq import SUPPORTED_TCONV_OP
import brevitas.nn as qnn
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
            self, layer, name, act_order, len_parallel_layers, create_weight_orig,
            num_blocks) -> None:
        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig)

        # Define how many columns to update in each mini-block
        self.blocksize = math.ceil(self.columns / num_blocks)

        # Initialize Hessian matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device='cpu',
                             dtype=torch.float32,
                             pin_memory=torch.cuda.is_available())
        self.B = torch.zeros((self.groups, self.columns, self.columns),
                             device='cpu',
                             dtype=torch.float32,
                             pin_memory=torch.cuda.is_available())
        self.nsamples = 0

        assert torch_version >= version.parse('1.10'), "GPTQ requires torch 1.10 or higher"

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        inp = self.process_input(input)
        batch_size = inp.shape[0]

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
            for i, inp in enumerate(inp_by_group):
                inp = unfold(inp)
                inp = inp.transpose(1, 0)
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        # Hessian computation
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.to(torch.float32)
        # optimizing CPU to GPU transfer using in-place copy to pinned memory
        self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
        self.H += self.B
        # If we are executing GPTQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self, percdamp=.01, c=1e4):
        assert not self.layer.weight_quant.requires_quant_input, "Error: GPTQ does not support weight quantizers that require quantized inputs."
        if hasattr(self.layer, 'allocate_params'):
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
                diag = torch.arange(self.columns, device='cpu')
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                # stabilizing the Cholesky decomposition with a fairly large constant, c
                self.H[i, :, :] = torch.linalg.cholesky(
                    self.H[i, :, :] * c, upper=True) / math.sqrt(c)
            h_inv = self.H
        except LinAlgError as e:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H, self.B

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            error_block = torch.zeros_like(
                weight[:, :, perm[i1:i2]], dtype=torch.float32)  # [groups, OC/groups, i2-i1]

            h_inv_block = h_inv[:, i1:i2, i1:i2]
            for i in range(count):
                q_groups = self.get_quant_weights(i, i1, permutation_list)  # [groups, OC/groups]
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

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2,
                                                          i2:].to(dev))).to(dtype)
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)


class A2GPTQ(AXE, GPTQ):
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
        assert self.max_accumulator_tile_size > 1, \
            "Error: accumulator tile size needs to be bigger than 1."
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
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
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
                    s = scales[group_index, bx].to(dtype)
                    n = neg_limits[group_index, bx]
                    p = pos_limits[group_index, bx]
                    q_arg = weight[group_index, :, perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
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


class gptq_mode(gpxq_mode):
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323
    Or accumulator-aware GPTQ (A2GPTQ) algorithm https://arxiv.org/pdf/2409.17092

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
        a2q_layer_filter_fnc (Callable[[Module], bool]): A function that takes a layer and returns True
            if the layer should be quantized with A2GPTQ. Default: lambda x: True
        max_accumulator_bit_width (Optional[int]): The maximum bit width of the accumulator. Default: None
        max_accumulator_tile_size (Optional[int]): The maximum tile size for accumulator-aware quantization.
            If `None` and `max_accumulator_bit_width` is specified, then a monolithic accumulator is
            assumed (see `Accumulator-Aware Post-Training Quantization` for more details). Default: None

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
            model: Module,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            num_blocks: int = 100,
            return_forward_output: bool = False,
            act_order: bool = False,
            gptq_class: GPTQ = GPTQ,
            a2q_layer_filter_fnc: Optional[Callable[[Module], bool]] = lambda x: True,
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

        # How many subblock to use during GPTQ for each layer
        self.num_blocks = num_blocks
        self.gptq_class = gptq_class
        self.max_accumulator_bit_width = max_accumulator_bit_width
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc

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

    def requires_accumulator_awareness(self, layer):
        # if the accumulator bit width is specified and the layer determined by the filter
        # then quantize with A2GPTQ
        if (self.max_accumulator_bit_width is not None) and self.a2q_layer_filter_fnc(layer):
            return True
        return False

    def initialize_module_optimizer(
            self, layer, name, act_order, len_parallel_layers, create_weight_orig):
        if self.requires_accumulator_awareness(layer):
            return A2GPTQ(
                layer=layer,
                name=name,
                act_order=act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                num_blocks=self.num_blocks,
                max_accumulator_bit_width=self.max_accumulator_bit_width,
                max_accumulator_tile_size=self.max_accumulator_tile_size)
        return self.gptq_class(
            layer=layer,
            name=name,
            act_order=act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            num_blocks=self.num_blocks)
