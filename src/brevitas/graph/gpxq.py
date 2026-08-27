# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from operator import attrgetter
from time import perf_counter
from typing import List
from typing import Optional
from typing import Set
import warnings

import torch
from torch.fx import GraphModule as TorchGraphModule
import torch.nn as nn
from tqdm import tqdm
import unfoldNd

from brevitas.fx import GraphModule
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.functional_quant import FunctionalLinearObservation
from brevitas.graph.functional_quant import FunctionalLinearTarget
from brevitas.graph.functional_quant import FunctionalLinearTargetBatch
from brevitas.graph.functional_quant import FunctionalQuantState
from brevitas.graph.utils import get_batch_dim
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import is_quant_module
import brevitas.nn as qnn
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.torch_utils import rename_tensor

SUPPORTED_CONV_OP = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0
    stop_forward: bool = True


class FunctionalGPxQBatch:
    """Shared invariants and quantization for one compatible functional expert batch."""

    def __init__(self, optimizers) -> None:
        if not optimizers:
            raise ValueError('A functional GPxQ batch cannot be empty.')
        first = optimizers[0]
        first_target = first.layer
        if not isinstance(first_target, FunctionalLinearTarget):
            raise TypeError('Functional GPxQ batching requires functional linear targets.')
        for optimizer in optimizers:
            target = optimizer.layer
            if optimizer.groups != 1:
                raise ValueError('Functional GPxQ expert targets must have one matrix group.')
            if not isinstance(target, FunctionalLinearTarget) or target.owner_id != first_target.owner_id or \
                    target.transpose_weight != first_target.transpose_weight:
                raise ValueError('Functional GPxQ batches require one owner and matrix layout.')
            if target.weight.shape != first_target.weight.shape or target.weight.dtype != first_target.weight.dtype or \
                    target.weight.device != first_target.weight.device:
                raise ValueError(
                    'Functional GPxQ batches require matching shape, dtype, and device.')
        self.optimizers = list(optimizers)
        self.targets = [optimizer.layer for optimizer in optimizers]
        self.weight = torch.stack([target.weight.detach() for target in self.targets])
        self._quantizers = {}

    def quantize(self, targets, weight):
        """Use proven row-separable groupwise batching, otherwise quantize targets locally."""
        if all(getattr(target.owner.proxy, 'is_groupwise', False) for target in targets):
            key = tuple(target.name for target in targets)
            if key not in self._quantizers:
                self._quantizers[key] = FunctionalLinearTargetBatch(targets, weight)
            return self._quantizers[key].quant_weight(weight)
        return torch.stack([target.quantize(weight[index]) for index, target in enumerate(targets)])

    def pop_buffer(self, name: str, device=None):
        """Stack and release one per-expert calibration buffer."""
        first_buffer = getattr(self.optimizers[0], name)
        output_device = first_buffer.device if device is None else device
        result = torch.empty((len(self.optimizers), *first_buffer.shape[1:]),
                             dtype=first_buffer.dtype,
                             device=output_device)
        for index, optimizer in enumerate(self.optimizers):
            result[index].copy_(getattr(optimizer, name)[0].to(output_device))
            delattr(optimizer, name)
        return result

    @staticmethod
    def writeback(targets, weight):
        """Write finite destination-dtype values and return targets that require fallback."""
        failed = []
        for index, target in enumerate(targets):
            value = weight[index].to(target.weight.dtype)
            if torch.isfinite(value).all():
                target.writeback(value)
            else:
                failed.append(target)
        return failed


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
            dtype: torch.dtype = torch.float32,
            functional_state: Optional[FunctionalQuantState] = None,
            min_samples: int = 0,
            insufficient_samples: str = 'rtn',
            expert_batch_size: int = 1) -> None:
        if functional_state is not None and not inplace:
            raise ValueError(
                'Functional GPxQ requires inplace=True because targets own model parameters.')
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
        self.functional_layer = LayerHandler(stop_forward=False)
        # How many layer to optimize
        self.num_layers = 0
        # Quantize following magnitude of activation
        self.act_order = act_order
        # the device and dtype of the buffers
        self.device = device
        self.dtype = dtype

        self.group_of_parallel_layers = group_of_parallel_layers
        self.return_forward_output = return_forward_output
        if min_samples < 0:
            raise ValueError('min_samples must be non-negative.')
        if expert_batch_size < 1:
            raise ValueError('expert_batch_size must be positive.')
        if insufficient_samples not in ('rtn', 'error', 'gpxq'):
            raise ValueError("insufficient_samples must be 'rtn', 'error', or 'gpxq'.")
        self.functional_state = functional_state
        self.min_samples = min_samples
        self.insufficient_samples = insufficient_samples
        self.expert_batch_size = expert_batch_size
        self.functional_targets = []
        self.functional_target_groups = []
        self.functional_collection_seconds = {}
        self.functional_observer_handle = None
        self.active_functional_group = None
        self.completed_functional_owners = set()

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
        if self.functional_state is not None:
            self.functional_targets = self.functional_state.iter_linear_targets(self.model)
            target_groups = {}
            for target in self.functional_targets:
                target_groups.setdefault(target.owner_id, []).append(target)
            self.functional_target_groups = list(target_groups.values())
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
                hooks = tuple(module._forward_hooks.values()) + tuple(
                    module._forward_pre_hooks.values())
                if any(not getattr(hook, '_brevitas_functional_quantization_hook', False)
                       for hook in hooks):
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

        for group in self.functional_target_groups:
            for target in group:
                self.gpxq_layers[target.name] = self.initialize_module_optimizer(
                    target,
                    target.name,
                    len_parallel_layers=1,
                    create_weight_orig=self.create_weight_orig)

        self.num_layers = len(dict_of_layers) + len(self.functional_target_groups)
        if self.functional_target_groups:
            self.functional_observer_handle = self.functional_state.register_linear_observer(
                self._observe_functional_target)
            # Ordinary module hooks stop calibration before later functional calls.
            # Defer expert scheduling until those module targets are exhausted.
            self._advance_functional_target()
        return self

    def __exit__(self, type, value, traceback):
        # Restore original quantization configuration
        super().__exit__(type, value, traceback)
        if self.functional_observer_handle is not None:
            self.functional_observer_handle.remove()
        for handle in self.hook_dict.values():
            handle.remove()
        self.hook_dict.clear()
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = self.orig_forward
        else:
            self.model.forward = self.orig_forward

    def update(self):
        for name in tuple(self.current_layer.layer_names):
            self.gpxq_layers[name].single_layer_update()
            handle = self.hook_dict.pop(name, None)
            if handle is not None:
                handle.remove()
        self.current_layer.layer_names.clear()

        if self.active_functional_group is not None:
            self._update_functional_group(self.active_functional_group)

        self._advance_functional_target()

    def _observe_functional_target(self, observation: FunctionalLinearObservation) -> None:
        """Collect routed activations for every expert in the scheduled owner."""
        if self.active_functional_group is None or observation.target.owner_id != self.active_functional_group[
                0].owner_id:
            return
        optimizer = self.gpxq_layers[observation.target.name]
        start = perf_counter()
        optimizer.update_batch(observation.target, (observation.input,), self.functional_layer)
        owner_id = observation.target.owner_id
        self.functional_collection_seconds[owner_id] = self.functional_collection_seconds.get(
            owner_id, 0.) + perf_counter() - start

    @property
    def active_functional_target(self) -> Optional[FunctionalLinearTarget]:
        """Expose the active owner's first target for existing callback compatibility."""
        if self.active_functional_group is None:
            return None
        return self.active_functional_group[0]

    def _update_functional_group(self, group) -> None:
        """Update all expert matrices of one owner after a shared calibration sweep."""
        owner_id = group[0].owner_id
        required_samples = max(1, self.min_samples)
        insufficient = [(target, self.gpxq_layers[target.name])
                        for target in group
                        if self.gpxq_layers[target.name].nsamples < required_samples]
        if insufficient and self.insufficient_samples == 'error':
            details = ', '.join(
                f'{target.name} has {optimizer.nsamples} samples' for target,
                optimizer in insufficient)
            raise RuntimeError(
                f'Functional GPxQ owner {owner_id} has insufficient calibration samples: {details}.'
            )

        insufficient_names = {target.name for target, _ in insufficient}
        update_start = perf_counter()
        progress = tqdm(
            total=len(group),
            desc=f'GPxQ {owner_id}',
            unit='expert',
            leave=False,
            disable=len(group) < 8)
        update_targets = []
        fallback_count = 0
        sample_fallback_targets = []
        for target in group:
            if target.name in insufficient_names and self.insufficient_samples != 'gpxq':
                optimizer = self.gpxq_layers[target.name]
                self._finish_functional_target(
                    target, optimizer, 'insufficient calibration samples', warn=False)
                sample_fallback_targets.append(target)
                fallback_count += 1
                progress.update()
            else:
                update_targets.append(target)
        if sample_fallback_targets:
            names = ', '.join(target.name for target in sample_fallback_targets[:8])
            remainder = len(sample_fallback_targets) - 8
            suffix = f', and {remainder} more' if remainder > 0 else ''
            warnings.warn(
                f'Functional GPxQ owner {owner_id} uses RTN fallback for '
                f'{len(sample_fallback_targets)} insufficiently calibrated experts: {names}{suffix}.',
                UserWarning)
        fallback_count += self._update_functional_targets(update_targets, progress)
        progress.close()

        collection_seconds = self.functional_collection_seconds.pop(owner_id, 0.)
        tqdm.write(
            f'Functional GPxQ {owner_id}: {len(group)} experts, '
            f'{len(group) - fallback_count} optimized, {fallback_count} fallback, '
            f'collection {collection_seconds:.1f}s, update {perf_counter() - update_start:.1f}s.')

        self.completed_functional_owners.add(owner_id)
        self.functional_layer.layer_names.clear()
        self.functional_layer.forward_count = 0
        self.active_functional_group = None

    def _update_functional_targets(self, targets, progress) -> int:
        """Apply the algorithm to functional targets, returning numerical fallbacks."""
        failed = 0
        for target in targets:
            optimizer = self.gpxq_layers[target.name]
            target_start = perf_counter()
            failed += int(optimizer.single_layer_update() is True)
            progress.set_postfix(
                samples=optimizer.nsamples, seconds=f'{perf_counter() - target_start:.1f}')
            progress.update()
        return failed

    def _finish_functional_target(
            self,
            target: FunctionalLinearTarget,
            optimizer: 'GPxQ',
            reason: str,
            warn: bool = True) -> None:
        """Release one target and retain ordinary proxy quantization on fallback."""
        if warn:
            warnings.warn(
                f'Functional GPxQ target {target.name} uses RTN fallback: {reason}.', UserWarning)
        optimizer.discard_calibration_buffers()

    def _advance_functional_target(self) -> None:
        """Move the observer to the next functional owner after each update cycle."""
        if self.hook_dict:
            return
        self.active_functional_group = next((
            group for group in self.functional_target_groups
            if group[0].owner_id not in self.completed_functional_owners),
                                            None)

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

        if create_weight_orig and not isinstance(
                self.layer, FunctionalLinearTarget) and not hasattr(self.layer, 'weight_orig'):
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

    def discard_calibration_buffers(self) -> None:
        """Release algorithm calibration state when a functional target falls back."""
        for name in ('H', 'G', 'B'):
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, 'quant_input'):
            self.quant_input = None

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
        if isinstance(self.layer, FunctionalLinearTarget):
            inp = _unpack_quant_tensor(inp)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)
            return inp.reshape(-1, inp.shape[-1]).t().unsqueeze(0)
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

    def get_quant_weights(self, i, i1, permutation_list, with_quant_history=False):

        if isinstance(self.layer, FunctionalLinearTarget):
            quant_weight = self.layer.quant_weight().unsqueeze(0)
            i = i1 + i
            if with_quant_history:
                return quant_weight[:, :, permutation_list[0][:i]]
            index = permutation_list[0][i]
            return quant_weight[:, :, index:index + 1].squeeze(2)

        # If the weight quantizer has not been initialized, raise an error
        for m in self.layer.weight_quant.modules():
            if hasattr(m, 'init_done') and not m.init_done:
                raise RuntimeError(
                    "Weight quantizer not initialized. Run a forward pass after quantization and try again."
                )

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
