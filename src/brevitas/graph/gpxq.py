# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from operator import attrgetter
from typing import List, Optional, Set
import warnings

import torch

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
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            act_order: bool = False,
            return_forward_output: bool = False) -> None:

        if not inplace:
            model = deepcopy(model)
        self.model = model
        self.create_weight_orig = create_weight_orig
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
                    gpxq_module_optimizer = self.initialize_module_optimizer(
                        module,
                        name,
                        act_order=self.act_order,
                        len_parallel_layers=len(parallel_layers),
                        create_weight_orig=self.create_weight_orig)
                    hook_fn = partial(
                        gpxq_module_optimizer.update_batch, current_layer=self.current_layer)
                    self.hook_dict[name] = module.register_forward_pre_hook(hook_fn)
                    self.gpxq_layers[name] = gpxq_module_optimizer
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


class GPxQ(ABC):

    def __init__(
            self, layer, name, act_order, len_parallel_layers=1, create_weight_orig=True) -> None:
        self.layer = layer
        self.name = name
        self.act_order = act_order

        weight = layer.weight.data

        if create_weight_orig and not hasattr(self.layer, 'weight_orig'):
            self.layer.register_buffer('weight_orig', layer.weight.detach().clone())

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
        self.len_parallel_layers = len_parallel_layers

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
                    value=torch.empty(
                        1, dtype=self.layer.weight.dtype, device=self.layer.weight.device),
                    scale=inp.scale,
                    zero_point=inp.zero_point,
                    bit_width=inp.bit_width,
                    signed=inp.signed,
                    training=inp.training)
            inp = inp.value
        elif self.layer.is_input_quant_enabled:
            self.quant_input = QuantTensor(
                value=torch.empty(
                    1, dtype=self.layer.weight.dtype, device=self.layer.weight.device),
                scale=self.layer.quant_input_scale(),
                zero_point=self.layer.quant_input_zero_point(),
                bit_width=self.layer.quant_input_bit_width(),
                signed=self.layer.is_quant_input_signed,
                training=self.layer.training)

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
    def update_batch(self):
        pass

    @abstractmethod
    def single_layer_update(self):
        pass

    def get_quant_weights(self, i, i1, permutation_list):
        # We need to recompute quant weights at runtime since our float weights are being updated
        # Add offset in case of blockwise computation
        i = i1 + i
        # For QuantLinear and for some QuantConvolutional layers, we exploit the possibility
        # of quantizing only a subset of the entire matrix speeding up the computation of GPxQ
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
