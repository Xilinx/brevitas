# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from itertools import product
from operator import attrgetter
import sys
from time import perf_counter
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm
import unfoldNd

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.functional_quant import FunctionalInterceptor
from brevitas.graph.functional_quant import FunctionalOperation
from brevitas.graph.functional_quant import FunctionalWeightOwner
from brevitas.graph.functional_quant import FunctionalWeightSource
from brevitas.graph.functional_quant import grouped_mm_functions
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


class _FunctionalTargetQuantHolder(nn.Module):
    """Present one two-dimensional target matrix to a Brevitas weight proxy."""

    def __init__(self, weight: torch.Tensor, output_channel_dim: int) -> None:
        super().__init__()
        self.weight = weight
        self.bias = None
        self.output_channel_dim = output_channel_dim
        self.out_channels = weight.shape[output_channel_dim]


class FunctionalLinearTarget:
    """Adapt one matrix view of a functional weight owner to GPxQ.

    Functional models can store many logical linear weights in one parameter, for
    example ``[experts, out_features, in_features]``. A target selects one leading
    index tuple, presents its matrix in canonical ``[out, in]`` orientation, owns a
    view-local quantizer, caches the floating reference weight, and writes updates
    back to the corresponding slice of the shared owner parameter.
    """

    def __init__(
            self,
            owner_id: str,
            owner: FunctionalWeightOwner,
            view_indices: Tuple[int, ...],
            transpose_weight: bool) -> None:
        self.owner_id = owner_id
        self.owner = owner
        self.view_indices = view_indices
        self.transpose_weight = transpose_weight
        self.reference_weight = None
        self.reference_pass = False
        self.target_quant_holder = None
        self.target_quant_proxy = None
        self.target_quant_device = None

    @property
    def key(self) -> Tuple[str, Tuple[int, ...]]:
        return self.owner_id, self.view_indices

    @property
    def name(self) -> str:
        suffix = ''.join(f'[{index}]' for index in self.view_indices)
        return f'{self.owner_id}{suffix}'

    def _owner_view(self, value: torch.Tensor) -> torch.Tensor:
        for index in self.view_indices:
            value = value[index]
        return value

    @property
    def weight(self) -> torch.Tensor:
        weight = self._owner_view(self.owner.original_parameter)
        return weight.t() if self.transpose_weight else weight

    def _target_proxy(self, native_weight: torch.Tensor) -> nn.Module:
        """Build a 2-D proxy so GPxQ never requantizes sibling matrices."""
        if self.target_quant_proxy is None:
            owner_injector = self.owner.proxy.quant_injector
            target_di_kwargs = self._remap_owner_axes(len(self.view_indices), 0)
            self.target_quant_holder = _FunctionalTargetQuantHolder(
                native_weight, target_di_kwargs.get('output_channel_dim', 0))
            target_injector = owner_injector.let(**target_di_kwargs)
            self.target_quant_proxy = target_injector.proxy_class(
                self.target_quant_holder, target_injector).to(native_weight.device)
            self.target_quant_proxy.train(self.owner.proxy.training)
            self.target_quant_device = native_weight.device
        else:
            self.target_quant_holder.weight = native_weight
            self.target_quant_holder.out_channels = native_weight.shape[
                self.target_quant_holder.output_channel_dim]
            if self.target_quant_device != native_weight.device:
                self.target_quant_proxy.to(native_weight.device)
                self.target_quant_device = native_weight.device
        return self.target_quant_proxy

    def _remap_owner_axes(self, dropped_dims: int, added_dims: int) -> Dict[str, int]:
        """Map owner quantizer axes after replacing leading owner dimensions."""
        owner_injector = self.owner.proxy.quant_injector
        owner_rank = self.owner.original_parameter.dim()
        target_di_kwargs = {}
        for name in ('output_channel_dim', 'group_dim'):
            axis = getattr(owner_injector, name, None)
            if axis is None:
                continue
            axis = axis if axis >= 0 else owner_rank + axis
            if axis < dropped_dims:
                raise RuntimeError(
                    f"Functional target '{self.name}' cannot use owner {name} {axis} after expert indexing."
                )
            target_di_kwargs[name] = axis - dropped_dims + added_dims
        return target_di_kwargs

    def quant_weight(self) -> torch.Tensor:
        return self.quantize(self.weight)

    def quantize(self, canonical_weight: torch.Tensor) -> torch.Tensor:
        """Quantize one canonical target matrix with its independent proxy."""
        native_weight = canonical_weight.t() if self.transpose_weight else canonical_weight
        weight = self._target_proxy(native_weight)(native_weight)
        value = weight.value if isinstance(weight, QuantTensor) else weight
        return value.t() if self.transpose_weight else value

    @property
    def weight_quant(self) -> nn.Module:
        """Expose this target's local proxy through the GPxQ layer contract."""
        return self._target_proxy(self._owner_view(self.owner.original_parameter))

    @property
    def weight_orig(self) -> torch.Tensor:
        """Preserve this target's floating matrix before its first update."""
        if self.reference_weight is None:
            self.reference_weight = self.weight.detach().clone().cpu()
        return self.reference_weight

    def writeback(self, value: torch.Tensor) -> None:
        """Copy a canonical updated matrix into this target's owner slice."""
        native_value = value.t() if self.transpose_weight else value
        with torch.no_grad():
            self._owner_view(self.owner.original_parameter).copy_(native_value)


class FunctionalLinearTargetBatch:
    """Quantize a temporary batch of compatible functional targets.

    The scheduler supplies targets from one owner with matching matrix layout.
    Their canonical weights are flattened into a row-separable proxy invocation
    and reshaped back to one matrix per target.
    """

    def __init__(
            self, targets: List[FunctionalLinearTarget], canonical_weight: torch.Tensor) -> None:
        first = targets[0]
        self.targets = targets
        flat_weight = canonical_weight.flatten(0, 1)
        di_kwargs = {'output_channel_dim': 0}
        if first.owner.proxy.is_groupwise:
            di_kwargs['group_dim'] = 1
        self.holder = _FunctionalTargetQuantHolder(flat_weight, 0)
        owner_injector = first.owner.proxy.quant_injector
        batch_injector = owner_injector.let(**di_kwargs)
        self.proxy = batch_injector.proxy_class(self.holder, batch_injector).to(flat_weight.device)
        self.proxy.train(first.owner.proxy.training)

    def quant_weight(self, canonical_weight: torch.Tensor) -> torch.Tensor:
        flat_weight = canonical_weight.flatten(0, 1)
        self.holder.weight = flat_weight
        self.holder.out_channels = flat_weight.shape[0]
        quant_weight = self.proxy(flat_weight)
        value = quant_weight.value if isinstance(quant_weight, QuantTensor) else quant_weight
        return value.reshape_as(canonical_weight)


def _storage_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the tensor storage used when indexing a plain or quantized weight."""
    if isinstance(value, QuantTensor):
        return value._value_ if getattr(value, '_is_groupwise', False) else value.value
    return value


class _FunctionalGPxQSession(FunctionalInterceptor):
    """Collect functional GPxQ inputs while a scoped model forward executes.

    Parametrization hooks capture each owner's latest materialized weight. The
    interceptor then identifies indexed runtime views from storage layout and
    dispatches their activation tensors to the matching GPxQ target. Linear-style
    operations use argument 0 as activation and argument 1 as weight; grouped-MM
    additionally uses argument 2 as cumulative expert offsets. Quantized and
    floating reference passes share the session but carry distinct phase state.
    """

    def __init__(
            self,
            model: nn.Module,
            functional_source: FunctionalWeightSource,
            owners: List[FunctionalWeightOwner],
            callback: Callable[[str, Tuple[int, ...], torch.Tensor, bool], None],
            linear_functions: Sequence[Callable],
            matmul_functions: Sequence[Callable],
            grouped_functions: Sequence[Callable]) -> None:
        super().__init__(operation_registry=functional_source.operation_registry)
        self.model = model
        self.functional_source = functional_source
        self.owners = {owner.id: owner for owner in owners}
        self.callback = callback
        self.linear_functions = tuple(linear_functions)
        self.matmul_functions = tuple(matmul_functions)
        self.grouped_functions = tuple(grouped_functions)
        self.runtime_weights = {owner.id: None for owner in owners}
        self.scope_depth = 0
        self.materialization_depth = 0
        self.enabled = True
        self.reference_pass = False

    def clear_runtime_weights(self) -> None:
        """Discard materialized roots captured during the current pass."""
        for owner_id in self.runtime_weights:
            self.runtime_weights[owner_id] = None

    def begin_quantized_pass(self) -> None:
        """Start a pass whose observations represent quantized execution."""
        self.reference_pass = False
        self.clear_runtime_weights()

    def begin_reference_pass(self) -> None:
        """Start a pass whose observations represent floating reference execution."""
        self.reference_pass = True
        self.clear_runtime_weights()

    def _scope_pre_hook(self, module, args) -> None:
        if self.scope_depth == 0:
            self.begin_quantized_pass()
        self.scope_depth += 1

    def _scope_post_hook(self, module, args, output) -> None:
        self.scope_depth = max(0, self.scope_depth - 1)

    def _materialization_pre_hook(self, module, args) -> None:
        self.materialization_depth += 1

    def _materialization_post_hook(self, owner_id, module, args, output) -> None:
        try:
            if self.enabled and self.scope_depth and isinstance(output, torch.Tensor):
                self.runtime_weights[owner_id] = _storage_tensor(output)
        finally:
            self.materialization_depth = max(0, self.materialization_depth - 1)

    def _attach_hooks(self) -> None:
        self.hooks.append(self.model.register_forward_pre_hook(self._scope_pre_hook))
        self.hooks.append(self.model.register_forward_hook(self._scope_post_hook, always_call=True))
        for owner_id, owner in self.owners.items():
            self.hooks.append(
                owner.parametrization.register_forward_pre_hook(self._materialization_pre_hook))
            self.hooks.append(
                owner.parametrization.register_forward_hook(
                    partial(self._materialization_post_hook, owner_id), always_call=True))

    def _clear_interception_state(self) -> None:
        self.scope_depth = 0
        self.materialization_depth = 0
        self.clear_runtime_weights()

    @staticmethod
    def _view_indices(root: torch.Tensor, value: torch.Tensor) -> Optional[Tuple[int, ...]]:
        """Recover leading integer indices for a supported storage-sharing view."""
        if root.device != value.device:
            return None
        try:
            same_storage = root.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        except RuntimeError:
            return None
        if not same_storage:
            return None

        rank_delta = root.dim() - value.dim()
        if rank_delta < 0:
            return None
        same_layout = tuple(value.shape) == tuple(root.shape[rank_delta:]) and tuple(
            value.stride()) == tuple(root.stride()[rank_delta:])
        transposed_layout = root.dim() >= 2 and value.dim() >= 2 and tuple(
            value.shape) == (*root.shape[rank_delta:-2], root.shape[-1], root.shape[-2]) and tuple(
                value.stride()) == (
                    *root.stride()[rank_delta:-2], root.stride()[-1], root.stride()[-2])
        if not same_layout and not transposed_layout:
            return None

        offset = value.storage_offset() - root.storage_offset()
        indices = []
        for axis in range(rank_delta):
            stride = root.stride()[axis]
            if stride == 0 or offset % stride:
                return None
            index = offset // stride
            if not 0 <= index < root.shape[axis]:
                return None
            indices.append(index)
            offset -= index * stride
        return tuple(indices) if offset == 0 else None

    def _owner_view(self, value) -> Optional[Tuple[FunctionalWeightOwner, Tuple[int, ...]]]:
        """Resolve a runtime weight operand to one prepared owner and index tuple."""
        if not isinstance(value, torch.Tensor):
            return None
        value = _storage_tensor(value)
        for owner_id, owner in self.owners.items():
            runtime_weight = self.runtime_weights[owner_id]
            roots = ([runtime_weight] if runtime_weight is not None else []) + [
                owner.original_parameter]
            for root in roots:
                indices = self._view_indices(_storage_tensor(root), value)
                if indices is not None:
                    return owner, indices
        return None

    def _observe_linear(self, operation, args, kwargs) -> None:
        """Dispatch the activation for one indexed linear or right-matmul weight."""
        inp = operation.argument(args, kwargs, 0)
        weight = operation.argument(args, kwargs, 1)
        owner_view = self._owner_view(weight)
        if owner_view is not None and isinstance(inp, torch.Tensor):
            self._dispatch(owner_view[0].id, owner_view[1], inp)

    def _dispatch(self, owner_id, indices, inp) -> None:
        """Send one observation while both functional interceptors are suspended."""
        with self.functional_source.suspend_quantization(), self.suspend():
            self.callback(owner_id, indices, inp, self.reference_pass)

    def _observe_grouped(self, operation, args, kwargs) -> None:
        """Split a grouped-MM activation into non-empty per-target observations."""
        inp = operation.argument(args, kwargs, 0)
        weight = operation.argument(args, kwargs, 1)
        offsets = operation.argument(args, kwargs, 2)
        owner_view = self._owner_view(weight)
        if owner_view is None or not isinstance(inp, torch.Tensor) or not isinstance(offsets,
                                                                                     torch.Tensor):
            return

        owner, prefix = owner_view
        leading_dims = owner.original_parameter.shape[:-2]
        view_indices = [
            (*prefix, *suffix)
            for suffix in product(*(range(size) for size in leading_dims[len(prefix):]))]
        boundaries = [int(offset) for offset in offsets.detach().cpu().tolist()]
        start = 0
        for indices, end in zip(view_indices, boundaries):
            if end > start:
                self._dispatch(owner.id, indices, inp[start:end])
            start = end

    def _matches(self, operation, func, functions) -> bool:
        return any(
            func is candidate or
            operation.canonical is self.operation_registry.resolve(candidate).canonical
            for candidate in functions)

    def _intercept(self, operation: FunctionalOperation, func, types, args, kwargs):
        if self.enabled and self.scope_depth and not self.materialization_depth:
            if self._matches(operation, func, self.grouped_functions):
                self._observe_grouped(operation, args, kwargs)
            elif self._matches(operation, func, (*self.linear_functions, *self.matmul_functions)):
                self._observe_linear(operation, args, kwargs)
        return func(*args, **kwargs)


class FunctionalGPxQBatch:
    """Stack per-target GPxQ buffers and weights for a batched update.

    Targets are grouped by the scheduler, so this helper assumes compatible shape,
    dtype, device, owner, and canonical orientation. It centralizes quantization,
    buffer release, and finite-result writeback used by batched GPxQ algorithms.
    """

    def __init__(self, optimizers) -> None:
        self.optimizers = list(optimizers)
        self.targets = [optimizer.layer for optimizer in optimizers]
        self.weight = torch.stack([target.weight.detach() for target in self.targets])
        self._quantizers = {}

    def quantize(self, targets, weight):
        """Use proven row-separable groupwise batching, otherwise quantize targets locally."""
        if all(target.owner.proxy.is_groupwise for target in targets):
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
    """Coordinate layerwise GPxQ calibration and weight updates.

    Ordinary quantized modules are collected through forward hooks. When a
    ``FunctionalWeightSource`` is provided, matrix owners are converted into one
    target per leading-index combination and observed through a scoped functional
    session. Ordinary modules are scheduled first, followed by functional owners
    in discovery order.

    Args:
        model: Model or model subtree optimized by the mode.
        group_of_parallel_layers: Optional groups of module names calibrated and
            updated together.
        inplace: Apply updates to ``model``. Functional targets require ``True``.
        create_weight_orig: Preserve floating module weights for algorithms that
            require a reference copy.
        use_quant_activations: Leave activation quantizers enabled during collection.
        act_order: Process columns according to activation magnitude when supported.
        return_forward_output: Run an additional forward with GPxQ collection
            disabled and return its output.
        device: Device used for calibration buffers, or ``"same"`` where supported.
        dtype: Data type used for calibration buffers and update calculations.
        functional_state: Optional ``FunctionalWeightSource`` providing prepared
            functional owners and call-sequence control.
        min_samples: Minimum observations required before optimizing a functional
            target. At least one observation is always required.
        insufficient_samples: Policy for under-observed targets: ``"rtn"``,
            ``"error"``, or ``"gpxq"``.
        expert_batch_size: Number of compatible functional targets updated together.
        functional_linear_functions: Additional linear-style operations with
            activation argument 0 and weight argument 1 in ``[out, in]`` layout.
        functional_matmul_functions: Additional right-matrix operations with
            activation argument 0 and weight argument 1.
        functional_grouped_mm_functions: Additional grouped-MM operations using
            activation, weight, and cumulative offsets at arguments 0, 1, and 2.

    Example:
        >>> from brevitas.graph.gptq import gptq_mode
        >>> with torch.no_grad():
        >>>     with gptq_mode(model) as gpxq:
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
        functional_state: Optional[FunctionalWeightSource] = None,
        min_samples: int = 0,
        insufficient_samples: str = 'rtn',
        expert_batch_size: int = 1,
        functional_linear_functions: Sequence[Callable] = (),
        functional_matmul_functions: Sequence[Callable] = (),
        functional_grouped_mm_functions: Sequence[Callable] = ()
    ) -> None:
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
        self.functional_source = functional_state
        # Functional GPxQ operations use one activation at argument 0 and one weight at argument 1.
        self.functional_linear_functions = (
            torch.nn.functional.linear, *functional_linear_functions)
        self.functional_matmul_functions = (
            torch.matmul,
            torch.Tensor.matmul,
            torch.Tensor.__matmul__,
            *functional_matmul_functions)
        self.functional_grouped_mm_functions = (
            *grouped_mm_functions(), *functional_grouped_mm_functions)
        self.min_samples = min_samples
        self.insufficient_samples = insufficient_samples
        self.expert_batch_size = expert_batch_size
        self.functional_targets = []
        self.functional_targets_by_key = {}
        self.functional_target_groups = []
        self.functional_collection_seconds = {}
        self.functional_session = None
        self._functional_session_entered = False
        self.active_functional_group = None
        self.completed_functional_owners = set()

        self.orig_forward = self.model.forward
        self._instance_forward = self.model.__dict__.get('forward')
        self._had_instance_forward = 'forward' in self.model.__dict__
        self._forward_overridden = False

    def _override_forward(self):
        self.model.forward = self.catch_stopfwd
        self._forward_overridden = True

    def _restore_forward(self):
        if not self._forward_overridden:
            return
        if self._had_instance_forward:
            self.model.forward = self._instance_forward
        else:
            delattr(self.model, 'forward')
        self._forward_overridden = False

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

    def _functional_operation_matches(self, func, candidates) -> bool:
        return any(
            func is candidate or
            self.functional_source.operation_registry.resolve(candidate).canonical is func
            for candidate in candidates)

    def _setup(self):
        """Create module optimizers, functional targets, hooks, and observation state."""
        # The user can specify on which layers to apply gptq in parallel.
        # All the others will be executed sequentially
        dict_of_layers = {
            name: [(name, module)] for name,
            module in self.model.named_modules() if self._is_module_supported(module)}
        if self.functional_source is not None:
            self.functional_targets = []
            functional_owners = []
            owner_layouts = {}
            for owner in self.functional_source.iter_weight_owners(self.model):
                layouts = []
                for func, argument_index, operand_transposed in owner.parameter_uses:
                    if argument_index != 1:
                        continue
                    if self._functional_operation_matches(func, self.functional_linear_functions):
                        layouts.append(operand_transposed)
                    elif self._functional_operation_matches(
                            func,
                        (*self.functional_matmul_functions, *self.functional_grouped_mm_functions)):
                        layouts.append(not operand_transposed)
                if layouts:
                    if any(layout != layouts[0] for layout in layouts[1:]):
                        raise RuntimeError(
                            f"Functional parameter '{owner.id}' has incompatible matrix layouts.")
                    functional_owners.append(owner)
                    owner_layouts[owner.id] = layouts[0]
            for owner in functional_owners:
                # Functional matrix owners define every leading-index combination as a target.
                leading_dims = owner.original_parameter.shape[:-2]
                self.functional_targets.extend(
                    FunctionalLinearTarget(owner.id, owner, indices, owner_layouts[owner.id])
                    for indices in product(*(range(size) for size in leading_dims)))
            self.functional_targets_by_key = {
                target.key: target for target in self.functional_targets}
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
            self.functional_session = _FunctionalGPxQSession(
                self.model,
                self.functional_source,
                functional_owners,
                self._observe_functional_target,
                self.functional_linear_functions,
                self.functional_matmul_functions,
                self.functional_grouped_mm_functions)
            self.functional_session.__enter__()
            self._functional_session_entered = True
            # Ordinary module hooks stop calibration before later functional calls.
            # Defer expert scheduling until those module targets are exhausted.
            self._advance_functional_target()

    def __enter__(self):
        """Apply quantization status changes and install GPxQ runtime state."""
        status_entered = False
        try:
            # Disable quantization selectively
            super().__enter__()
            status_entered = True
            self._setup()
            self._override_forward()
        except Exception:
            exc_info = sys.exc_info()
            if self.functional_session is not None and self._functional_session_entered:
                self.functional_session.__exit__(*exc_info)
            self.functional_session = None
            self._functional_session_entered = False
            for handle in self.hook_dict.values():
                handle.remove()
            self.hook_dict.clear()
            self._restore_forward()
            if status_entered:
                super().__exit__(*exc_info)
            raise
        return self

    def __exit__(self, type, value, traceback):
        """Remove sessions and hooks, restore ``forward``, and reset quantization status."""
        try:
            if self.functional_session is not None and self._functional_session_entered:
                self.functional_session.__exit__(type, value, traceback)
            self.functional_session = None
            self._functional_session_entered = False
            return super().__exit__(type, value, traceback)
        finally:
            for handle in self.hook_dict.values():
                handle.remove()
            self.hook_dict.clear()
            self._restore_forward()

    def update(self):
        """Update the currently scheduled module group or functional owner."""
        for name in tuple(self.current_layer.layer_names):
            self.gpxq_layers[name].single_layer_update()
            handle = self.hook_dict.pop(name, None)
            if handle is not None:
                handle.remove()
        self.current_layer.layer_names.clear()

        if self.active_functional_group is not None:
            self._update_functional_group(self.active_functional_group)

        self._advance_functional_target()

    def _observe_functional_target(
            self,
            owner_id: str,
            view_indices: Tuple[int, ...],
            input: torch.Tensor,
            reference_pass: bool) -> None:
        """Collect routed activations for every expert in the scheduled owner."""
        if self.active_functional_group is None or owner_id != self.active_functional_group[
                0].owner_id:
            return
        target = self.functional_targets_by_key.get((owner_id, view_indices))
        if target is None:
            return
        optimizer = self.gpxq_layers[target.name]
        start = perf_counter()
        previous_reference_pass = target.reference_pass
        target.reference_pass = reference_pass
        try:
            optimizer.update_batch(target, (input,), self.functional_layer)
        finally:
            target.reference_pass = previous_reference_pass
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
    """Base optimizer for one module or functional matrix target.

    Subclasses allocate algorithm-specific calibration buffers, consume forward
    observations through :meth:`update_batch`, and update the target weight in
    :meth:`single_layer_update`. Returning ``True`` from the update signals that
    numerical failure required the caller to retain the fallback quantization.
    """

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
    def update_batch(self, *args, **kwargs):
        """Accumulate calibration statistics from one observed forward input."""
        raise NotImplementedError

    @abstractmethod
    def single_layer_update(self, *args, **kwargs):
        """Apply one algorithm update and report whether fallback was required."""
        raise NotImplementedError

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
