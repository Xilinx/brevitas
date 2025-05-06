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
from torch.fx import GraphModule as TorchGraphModule
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import is_quant_module
import brevitas.nn as qnn
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor

SUPPORTED_CONV_OP = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0


class gpxq_mode(quantization_status_manager):
    """
    Apply GPxQ algorithm.

    Args:
        model (Module): The model to quantize with GPxQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPFQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gpxq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPxQ. Default: False
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with gpxq_mode(model) as gpxq:
        >>>         gpxq_mode = gpxq.model
        >>>         for i in tqdm(range(gpxq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gpxq_mode(img)
        >>>             gpxq.update()
    """

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
        # Note that if use_quant_activations = True, the super() context manager
        # is equivalent to a nullcontext
        super().__init__(
            model=model,
            disable_act_quant=not use_quant_activations,
            disable_weight_quant=False,
            disable_bias_quant=not use_quant_activations,
        )
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

        self.group_of_parallel_layers = group_of_parallel_layers
        self.return_forward_output = return_forward_output

        self.orig_forward = self.model.forward
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = self.catch_stopfwd
        else:
            self.model.forward = self.catch_stopfwd

    def _is_module_supported(self, module):
        if is_quant_module(module):
            is_quant_enabled = module.weight_quant.is_quant_enabled
        else:
            is_quant_enabled = False
        if isinstance(module, (nn.Linear, *SUPPORTED_CONV_OP)):
            return is_quant_enabled
        else:
            return False

    def __enter__(self):
        # Disable quantization selectively
        super().__enter__()
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
                        len_parallel_layers=len(parallel_layers),
                        create_weight_orig=self.create_weight_orig)
                    hook_fn = partial(
                        gpxq_module_optimizer.update_batch, current_layer=self.current_layer)
                    self.hook_dict[name] = module.register_forward_pre_hook(hook_fn)
                    self.gpxq_layers[name] = gpxq_module_optimizer

        self.num_layers = len(dict_of_layers)
        return self

    def __exit__(self, type, value, traceback):
        # Restore original quantization configuration
        super().__exit__(type, value, traceback)
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = self.orig_forward
        else:
            self.model.forward = self.orig_forward

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
        self.create_weight_orig = create_weight_orig

        weight_shape = torch.tensor(layer.weight.shape)

        if create_weight_orig and not hasattr(self.layer, 'weight_orig'):
            self.layer.register_buffer('weight_orig', layer.weight.detach().clone())

        # By default, use groups = 1
        self.groups = 1
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight_shape[1], weight_shape[0] = weight_shape[0], weight_shape[1]
            self.groups = self.layer.groups

        # Number of rows is equal to the output channels (OC)
        self.rows = weight_shape[0]
        # Number of columns is equal to the input channels (IC)
        self.columns = torch.prod(weight_shape[1:])
        self.len_parallel_layers = len_parallel_layers

        self.disable_pre_forward_hook = False
        # Some layers require knowledge from quant inputs to compute quant weights
        self.quant_metadata = None

    def process_input(self, inp):
        # Input is a tuple, so we take first element
        inp = inp[0]
        if is_quant_module(self.layer):
            inp = self.layer.input_quant(inp)
            is_quant_enabled = self.layer.weight_quant.is_quant_enabled
        else:
            is_quant_enabled = False

        # If using quantized activations, inp could be QuantTensor. In
        # this case, we overwrite the metadata.
        if isinstance(inp, QuantTensor):
            if is_quant_enabled and self.quant_metadata is None:
                self.quant_metadata = self.layer.input_quant.cache_class(inp, metadata_only=True)
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
    def update_batch(self):
        pass

    @abstractmethod
    def single_layer_update(self):
        pass

    def get_quant_weights(self, i, i1, permutation_list, with_quant_history=False):

        # We need to recompute quant weights at runtime since our float weights are being updated
        # Add offset in case of blockwise computation
        i = i1 + i

        # For QuantLinear and for some QuantConvolutional layers, we exploit the possibility
        # of quantizing only a subset of the entire matrix speeding up the computation of GPxQ
        no_slice = False
        # Groupwise Quantization does not support slicing
        no_slice = no_slice or self.layer.weight_quant.is_groupwise
        # If we need quantization of past channels, we do not use slicing
        no_slice = no_slice or with_quant_history
        # If we are in export mode (i.e., inference mode), we do not slice for torch.compile
        # compatibility
        no_slice = no_slice or self.layer.weight_quant.export_mode

        if isinstance(self.layer, qnn.QuantLinear):
            if no_slice:

                # No slicing, not optimized
                q = self.layer.quant_weight(quant_input=self.quant_metadata)
                q = _unpack_quant_tensor(q).unsqueeze(0)  # [1, OC, IC]
                if with_quant_history:
                    return q[:, :, permutation_list[0][:i]]  # [1, OC, i]
                index = permutation_list[0][i]  # only 1 group for linear layers
                q = q[:, :, index:index + 1]  # [1, OC, 1]
            else:
                index = permutation_list[0][i]
                subtensor_slice_list = [None, (index, index + 1)]
                q = _unpack_quant_tensor(
                    self.layer.quant_weight(
                        subtensor_slice_list=subtensor_slice_list,
                        quant_input=self.quant_metadata)).unsqueeze(0)  # [1, OC, 1]
        elif isinstance(self.layer, SUPPORTED_CONV_OP):
            # Depthwise and ConvTranspose does not support slicing
            no_slice_conv = no_slice or (self.groups > 1 or is_conv_transposed(self.layer))

            if no_slice_conv:

                quant_weight = self.layer.quant_weight(quant_input=self.quant_metadata)
                quant_weight = _unpack_quant_tensor(quant_weight)

                if is_conv_transposed(self.layer):
                    quant_weight = quant_weight.transpose(1, 0)  # This performs a view
                quant_weight = quant_weight.flatten(1)
                quant_weight = quant_weight.view(self.groups, -1, quant_weight.shape[-1])

                if self.act_order:
                    for ii, perm in enumerate(permutation_list):
                        quant_weight[ii, :, :] = quant_weight[ii, :, perm]

                if with_quant_history:
                    return quant_weight[:, :, :i]  # [groups, OC/groups, i]
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
                q = _unpack_quant_tensor(
                    self.layer.quant_weight(
                        subtensor_slice_list=index_2d_to_nd,
                        quant_input=self.quant_metadata)).flatten(1)  # [OC, 1]
                q = q.unsqueeze(0)  # [1, OC, 1]
        # We need to remove the last dim
        q = q.squeeze(2)  # [groups, OC/groups] or [1, OC]
        return q
