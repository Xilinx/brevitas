# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from operator import attrgetter
from typing import List
from typing import Optional
from typing import Set
import warnings

import torch
from torch.fx import GraphModule as TorchGraphModule
import torch.nn as nn
import unfoldNd

from brevitas.fx import GraphModule
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.utils import get_batch_dim
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import is_quant_module
import brevitas.nn as qnn
from brevitas.nn.mixin import WeightRegion
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.torch_utils import rename_tensor

SUPPORTED_CONV_OP = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0


class GPxQWeightReader:
    """Read quantized weights in GPxQ's ``[groups, rows, columns]`` layout."""

    def __init__(self, owner):
        self.owner = owner

    def _check_initialized(self):
        for module in self.owner.layer.weight_quant.modules():
            if hasattr(module, 'init_done') and not module.init_done:
                raise RuntimeError(
                    "Weight quantizer not initialized. Run a forward pass after quantization and try again."
                )

    def _full(self):
        quant_weight = self.owner.layer.quant_weight(quant_input=self.owner.quant_metadata)
        quant_weight = _unpack_quant_tensor(quant_weight)
        if isinstance(self.owner.layer, qnn.QuantLinear):
            return quant_weight.unsqueeze(0)
        if is_conv_transposed(self.owner.layer):
            quant_weight = quant_weight.transpose(1, 0)
        quant_weight = quant_weight.flatten(1)
        return quant_weight.view(self.owner.groups, -1, quant_weight.shape[-1])

    def current(self, position, permutation_list):
        """Return one current quantized column as ``[groups, rows, 1]``."""
        self._check_initialized()
        if isinstance(self.owner.layer, qnn.QuantLinear):
            index = int(permutation_list[0][position])
            region = WeightRegion((None, (index, index + 1)))
            quant_weight = self.owner.layer.quant_weight_region(
                region=region, quant_input=self.owner.quant_metadata)
            return quant_weight.unsqueeze(0)

        quant_weight = self._full()
        columns = []
        for group_index, permutation in enumerate(permutation_list):
            index = permutation[position]
            columns.append(quant_weight[group_index, :, index:index + 1])
        return torch.stack(columns)

    def history(self, end, permutation_list):
        """Return the quantized permutation prefix as ``[groups, rows, end]``."""
        self._check_initialized()
        quant_weight = self._full()
        history = []
        for group_index, permutation in enumerate(permutation_list):
            history.append(quant_weight[group_index].index_select(1, permutation[:end]))
        return torch.stack(history)


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
        device (str): Device the buffers are stored on. Default: cpu
        dtype (torch.dtype): Datatype the buffers are stored in. Default: torch.float32

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
            return_forward_output: bool = False,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32) -> None:
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
        # the device and dtype of the buffers
        self.device = device
        self.dtype = dtype

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
            # ConvTranspose is temporarily unsupported in GPxQ
            # See https://github.com/Xilinx/brevitas/issues/1479
            if is_conv_transposed(module):
                warnings.warn("ConvTranspose is temporarily unsupported for GPxQ, skipping.")
                return False
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
            self,
            layer,
            name,
            act_order,
            len_parallel_layers=1,
            create_weight_orig=True,
            device='cpu',
            dtype=torch.float32) -> None:
        self.layer = layer
        self.name = name
        self.act_order = act_order
        self.create_weight_orig = create_weight_orig
        # device and dtype of buffers; 'same' means using the same device for the buffer as the layer weights
        self.device = layer.weight.device if device == 'same' else device
        self.dtype = dtype

        weight_shape = torch.tensor(layer.weight.shape)

        if create_weight_orig and not hasattr(self.layer, 'weight_orig'):
            self.layer.register_buffer('weight_orig', layer.weight.detach().clone().cpu())

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
        self.weight_reader = GPxQWeightReader(self)

    @property
    def use_intermediate_buffer(self):
        # By default, we are optimizing for minimizing peak memory usage, which is
        # when self.device=='cpu'. Since the compute is done on the GPU but the buffers
        # are on the GPU, we optimize the CPU to GPU transfer using in-place copy to
        # pinned memory in an intermediate buffer, usually self.B
        return self.device == 'cpu'

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

        # Define batch size before re-organizing the input. Prefer batch_dim/batch_first exposed
        # by the module; fall back to named tensors (PyTorch < 2.13).
        batch_dim = get_batch_dim(self.layer, inp)
        # Strip any legacy dimension names before reshaping (no-op on PyTorch >= 2.13).
        inp = rename_tensor(inp, None)
        if batch_dim:
            inp = inp.transpose(0, batch_dim)

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, nn.Linear):
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
            for i, inp in enumerate(inp_by_group):
                inp = unfold(inp)
                inp = inp.transpose(1, 0)
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        return inp_processed

    @abstractmethod
    def update_batch(self):
        pass

    @abstractmethod
    def single_layer_update(self):
        pass

    def get_quant_weight(self, i, i1, permutation_list):
        return self.weight_reader.current(i1 + i, permutation_list).squeeze(2)

    def get_quant_weight_history(self, end, permutation_list):
        return self.weight_reader.history(end, permutation_list)
