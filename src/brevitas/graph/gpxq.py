# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
import math
from operator import attrgetter
from typing import List, Optional, Set
import warnings

import torch

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError
import numpy as np
import unfoldNd

from brevitas.graph.calibrate import DisableEnableQuantization
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor

SUPPORTED_CONV_OP = (
    qnn.QuantConv2d, qnn.QuantConv1d, qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)


class StopFwdException(Exception):
    pass


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0


class gpxq_mode(ABC):

    def __init__(
            self,
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            use_quant_activations: bool = True,
            act_order: bool = False,
            return_forward_output: bool = False) -> None:

        if not inplace:
            model = deepcopy(model)
        self.model = model
        self.use_quant_activations = use_quant_activations
        self.hook_dict = dict()
        self.gpxq_layers = dict()
        # reference for each layer to update
        self.current_layer = LayerHandler()
        # How many layer to optimize
        self.num_layers = 0
        # Quantize following magnitude of activation
        self.act_order = act_order
        # How many subblock to use during GPTQ for each layer

        self.disable_quant_inference = DisableEnableQuantization()

        self.group_of_parallel_layers = group_of_parallel_layers
        self.return_forward_output = return_forward_output

    def _is_module_supported(self, module):
        if isinstance(module, SUPPORTED_CONV_OP):
            return True
        elif isinstance(module, qnn.QuantLinear):
            return True
        else:
            return False

    def __enter__(self):
        # The user can specify on which layers to apply gptq in parallel.
        # All the others will be executed sequentially
        dict_of_layers = {
            name: [(name, module)] for name,
            module in self.model.named_modules() if self._is_module_supported(module)}
        if self.group_of_parallel_layers is not None:
            for parallel_layers in self.group_of_parallel_layers:
                for name in parallel_layers:
                    if name not in dict_of_layers:
                        raise ValueError(
                            "The layer {} is not present in the model or it is not supported for GPTQ"
                            .format(name))
                    del dict_of_layers[name]
                names = '_'.join(parallel_layers)
                dict_of_layers[names] = [
                    (name, attrgetter(name)(self.model)) for name in parallel_layers]

        # Print warning if hooks are attached to any module, since the normal forward flow of the
        # network is highly disrupted during GPxQ
        for _, parallel_layers in dict_of_layers.items():
            for name, module in parallel_layers:
                if len(module._forward_hooks) > 0 or len(module._forward_pre_hooks):
                    warnings.warn(
                        f'Hooks detected during setup for GPxQ. '
                        f'Behaviour might deviate from what expected.')

                # Attach hooks for GPTQ
                if self._is_module_supported(module):
                    gpxq = self.class_implementation(
                        module, name, act_order=self.act_order, parallel_layers=parallel_layers)
                    hook_fn = partial(gpxq.update_batch, current_layer=self.current_layer)
                    self.hook_dict[name] = module.register_forward_pre_hook(hook_fn)
                    self.gpxq_layers[name] = gpxq
        if not self.use_quant_activations:
            self.disable_quant_inference.disable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.disable_bias_quantization(
                self.model, is_training=self.model.training)

        self.num_layers = len(dict_of_layers)
        return self

    def __exit__(self, type, value, traceback):
        self.model.forward = self.orig_forward
        if not self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.enable_bias_quantization(
                self.model, is_training=self.model.training)

    def update(self):
        for name in self.current_layer.layer_names:
            self.gpxq_layers[name].single_layer_update()
            self.hook_dict[name].remove()
        self.current_layer.layer_names.clear()

    @abstractmethod
    def catch_stopfwd(self, *args, **kwargs):
        pass


class gptq_mode(gpxq_mode):
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323.

    Args:
        model (Module): The model to quantize with GPTQ
        inplace (bool): Wheter to apply GPTQ inplace or perform a deepcopy. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPTQ. Default: False

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
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            use_quant_activations: bool = True,
            num_blocks: int = 100,
            act_order: bool = False) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(model, group_of_parallel_layers, inplace, use_quant_activations, act_order)

        self.orig_forward = self.model.forward
        self.model.forward = self.catch_stopfwd
        # How many subblock to use during GPTQ for each layer
        self.num_blocks = num_blocks
        self.class_implementation = GPTQ
        GPTQ.num_blocks = num_blocks

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        finally:
            if self.return_forward_output:
                # If we want to return the output of the network, we need to disable all hooks
                for name, gptq_class in self.gptq_layers.items():
                    gptq_class.disable_pre_forward_hook = True
                out = self.orig_forward(*args, **kwargs)
                for name, gptq_class in self.gptq_layers.items():
                    gptq_class.disable_pre_forward_hook = False
                return out


class gpfq_mode(gpxq_mode):
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323.

    Args:
        model (Module): The model to quantize with GPTQ
        inplace (bool): Wheter to apply GPTQ inplace or perform a deepcopy. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPTQ. Default: False

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
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            use_quant_activations: bool = True,
            act_order: bool = False) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(model, group_of_parallel_layers, inplace, use_quant_activations, act_order)

        self.orig_forward = self.model.forward
        self.model.forward = self.catch_stopfwd
        self.class_implementation = GPFQ

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        # Before collecting float input, restore original float weights if they have been modified
        quant_weight = dict()
        for module in self.model.modules():
            if hasattr(module, 'weight_orig_data'):
                quant_weight[module] = deepcopy(module.weight.data)
                module.weight.data = module.weight_orig_data
        # Disable quantization
        self.disable_quant_inference.disable_param_quantization(self.model, is_training=False)
        self.disable_quant_inference.disable_act_quantization(self.model, is_training=False)
        # Collect float input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        # Restore correct weights
        for module in self.model.modules():
            if hasattr(module, 'weight_orig_data'):
                module.weight.data = quant_weight[module]
        # Re-enable quantization. If activation quantization is disabled,
        # we also disable bias quantization
        self.disable_quant_inference.enable_param_quantization(self.model, is_training=False)
        if self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(self.model, is_training=False)
        else:
            self.disable_quant_inference.disable_bias_quantization(self.model, is_training=False)


class GPxQ(ABC):

    def __init__(self, layer, name, act_order, parallel_layers=1) -> None:
        self.layer = layer
        self.name = name
        self.act_order = act_order

        weight = layer.weight.data
        self.layer.weight_orig_data = deepcopy(weight)
        # By default, use groups = 1
        self.groups = 1
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
            self.groups = self.layer.groups

        # Number of rows is equal to the output channels (OC)
        self.rows = weight.shape[0]
        # Number of columns is equal to the input channels (IC)
        self.columns = weight.shape[1]
        self.parallel_layers = parallel_layers

        self.disable_pre_forward_hook = False
        # Some layers require knowledge from quant inputs to compute quant weights
        self.quant_input = None

    
    def process_input(self, inp):
        # Input is a tuple, so we take first element
        inp = inp[0]
        # If using Quant Activations, inp could be QuantTensor
        if isinstance(inp, QuantTensor):
            if self.layer.weight_quant_requires_quant_input:
                # Can minimize memory allocation by not storing actual values
                self.quant_input = QuantTensor(
                    value=None,
                    scale=inp.scale,
                    zero_point=inp.zero_point,
                    bit_width=inp.bit_width,
                    signed=inp.signed,
                    training=inp.training)
            inp = inp.value

        # If input is unbatched, add batch_size = 1
        if len(inp.shape) == 1:
            warnings.warn("Found unbatched input, adding batch dimension equal to 1")
            inp = inp.unsqueeze(0)

        # Define batch size before re-organizing the input
        if hasattr(inp, 'names') and 'N' in inp.names:
            batch_dim = inp.names.index('N')
            inp.rename_(None)
            inp = inp.transpose(0, batch_dim)
        return inp

    @abstractmethod
    def update_batch(self, module, input, current_layer):
        pass

    @abstractmethod
    def single_layer_update(self, percdamp=.01):
        pass

    def get_quant_weights(self, i, i1, permutation_list):
        # We need to recompute quant weights at runtime since our float weights are being updated
        # Add offset in case of blockwise computation (e.g., GPTQ)
        i = i1 + i
        # For QuantLinear and for some QuantConvolutional layers, we exploit the possibility
        # of quantizing only a subset of the entire matrix speeding up the computation of GPTQ
        if isinstance(self.layer, qnn.QuantLinear):
            index = permutation_list[0][i]
            subtensor_slice_list = [None, (index, index + 1)]
            q = self.layer.quant_weight(
                subtensor_slice_list=subtensor_slice_list,
                quant_input=self.quant_input).value.unsqueeze(0)  # [1, OC, 1]
        elif isinstance(self.layer, SUPPORTED_CONV_OP):
            # For depthwise and ConvTranspose we fall back to quantizing the entire martix.
            # For all other cases, we create a mask that represent the slicing we will perform on the weight matrix
            # and we quantize only the selected dimensions.
            if self.groups > 1 or (self.groups == 1 and isinstance(
                    self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d))):

                quant_weight = self.layer.quant_weight(quant_input=self.quant_input)
                quant_weight = quant_weight.value

                if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                    quant_weight = quant_weight.transpose(1, 0)  # This performs a view
                quant_weight = quant_weight.flatten(1)
                quant_weight = quant_weight.view(self.groups, -1, quant_weight.shape[-1])

                if self.act_order:
                    for ii, perm in enumerate(permutation_list):
                        quant_weight[ii, :, :] = quant_weight[ii, :, perm]

                q = quant_weight[:, :, i:i + 1]  # [groups, OC/groups, 1]
            else:
                index = permutation_list[0][i]
                shapes = self.layer.weight.shape[1:]
                index_2d_to_nd = []
                residual_index = index.item()
                for shape in shapes[::-1]:
                    index_2d_to_nd.append((residual_index % shape, residual_index % shape + 1))
                    residual_index = residual_index // shape
                index_2d_to_nd = index_2d_to_nd[::-1]
                index_2d_to_nd.insert(0, None)
                q = self.layer.quant_weight(
                    subtensor_slice_list=index_2d_to_nd,
                    quant_input=self.quant_input).value.flatten(1)  # [OC, 1]
                q = q.unsqueeze(0)  # [1, OC, 1]
        # We need to remove the last dim
        q = q.squeeze(2)  # [groups, OC/groups] or [1, OC]
        return q


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
    num_blocks = 100

    def __init__(self, layer, name, act_order, parallel_layers=1) -> None:
        super().__init__(layer, name, act_order, parallel_layers)

        dev = self.layer.weight.device

        # Define how many columns to update in each mini-block
        self.blocksize = math.ceil(self.columns / GPTQ.num_blocks)

        # Initialize Hessian matrix and counter. We need it in float32 to compute the inverse
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device=dev,
                             dtype=torch.float32)
        self.nsamples = 0

    def update_batch(self, module, input, current_layer):
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
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
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
        self.H += inp_processed.bmm(inp_processed.transpose(2, 1))
        # If we are executing GPTQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == len(self.parallel_layers):
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self, percdamp=.01):
        weight = self.layer.weight.data
        dev = weight.device

        # Store the original dtype of the weights
        # During computation, everything is converted to float32.
        # When the weights are updated, we cast everything back to the original dtype
        dtype = weight.dtype

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
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
                diag = torch.arange(self.columns, device=dev)
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :], upper=True)
            h_inv = self.H
        except LinAlgError as e:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

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
                    q = q_groups[group_index]  # [OC/groups]
                    w = weight[group_index, :, perm[i1:i2][i]].to(torch.float32)  # [OC/groups]
                    d = h_inv_block[group_index, i, i]  # [1]
                    error = (w - q) / d  # [OC/groups]
                    error_block[group_index, :, i] = error
                    # We need to update the original weights
                    weight[group_index, :, perm[i1:i2][i:]] -= (
                        error.unsqueeze(1).matmul(h_inv_block[group_index, i,
                                                              i:].unsqueeze(0))).to(dtype)

            for group_index in range(self.groups):
                perm = permutation_list[group_index]
                weight[group_index, :, perm[i2:]] -= (
                    error_block[group_index].matmul(h_inv[group_index, i1:i2, i2:])).to(dtype)


class GPFQ(GPxQ):
    """
    Based on https://github.com/YixuanSeanZhou/Quantized_Neural_Nets/tree/main
    """

    def __init__(self, layer, name, act_order, parallel_layers=1) -> None:
        super().__init__(layer, name, act_order, parallel_layers)
        self.float_input = None
        self.quantized_input = None
        self.index_computed = False
        self.p = 0.25

    def update_batch(self, module, input, current_layer):

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        is_quant_disabled = module.weight_quant.disable_quant

        inp = self.process_input(input)
        batch_size = inp.shape[0]

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                unfold_impl = unfoldNd.UnfoldTransposeNd
            else:
                unfold_impl = unfoldNd.UnfoldNd

            unfold = unfold_impl(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.kernel_size)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for i, inp in enumerate(inp_by_group):

                inp = unfold(inp)

                batch_size, num_blocks = inp.shape[0], inp.shape[-1]
                inp = torch.transpose(inp, 1, 2)  # shape (B, L, C*kernel_size[0]*kernel_size[1])
                inp = inp.reshape(-1, inp.size(-1))  # shape (B*L, C*kernel_size[0]*kernel_size[1])

                if not self.index_computed:
                    self.index_computed = True
                    self.rand_indices = np.concatenate([
                        np.random.choice(
                            np.arange(num_blocks * i, num_blocks * (i + 1)),
                            size=int(
                                self.p * num_blocks + 1 if self.p != 1 else self.p * num_blocks))
                        for i in range(batch_size)])  # need to define self.p (probability)

                indexes = self.rand_indices
                if np.max(self.rand_indices) > inp.shape[0]:
                    indexes = self.rand_indices < inp.shape[0]
                    indexes = self.rand_indices[indexes]
                    

                inp = inp[indexes]
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        if is_quant_disabled:
            if self.float_input is None:
                self.float_input = inp_processed
            else:
                self.float_input = torch.cat([self.float_input, inp_processed], dim=1)
        else:
            if self.quantized_input is None:
                self.quantized_input = inp_processed
            else:
                self.quantized_input = torch.cat([self.quantized_input, inp_processed], dim=1)
        raise StopFwdException

    def single_layer_update(self):
        weight = self.layer.weight.data
        dev = weight.device
        dtype = weight.dtype
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1]) # [Groups, OC/Groups, IC]
        U = torch.zeros(weight.shape[0], weight.shape[1], self.float_input.shape[1], device=dev, dtype=dtype)

        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                U[group_index] += weight[group_index, :, t].unsqueeze(1) * self.float_input[group_index, :, t].unsqueeze(0) #[OC/Groups, 1] * [1, INSHAPE[1]]
                norm = torch.linalg.norm(self.quantized_input[group_index, :, t], 2) ** 2
                if norm > 0:
                    q_arg = U[group_index].matmul(self.quantized_input[group_index, :, t]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])

                weight[group_index, :, t] = q_arg
            q = self.get_quant_weights(t, 0, [torch.tensor(range(weight.shape[-1]))])
            for group_index in range(self.groups):
                U[group_index] -= q[group_index].unsqueeze(1) * self.quantized_input[group_index, :, t].unsqueeze(0)

        del self.float_input
        del self.quantized_input
