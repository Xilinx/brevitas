# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
import contextlib
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import warnings

from packaging import version
import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.parametrize import is_parametrized
from torch.nn.utils.parametrize import ParametrizationList
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrize import remove_parametrizations
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

from brevitas import torch_version
from brevitas.nn import QuantIdentity
from brevitas.quant_tensor import QuantTensor

# Runtime quantization for calls to torch functional operators.

__all__ = [
    'FunctionalQuantState',
    'functional_quantization_mode',
    'grouped_mm_functions',
    'prepare_functional_quantization',
    'remove_functional_quantization']

QuantResolverResult = Optional[Union[Type, Tuple[Optional[Type], Dict[str, Any]]]]
QuantResolver = Callable[[nn.Module, str, int], QuantResolverResult]
QuantResolvable = Optional[Union[Type, QuantResolver]]
QuantSpecElement = Union[QuantResolvable, Tuple[QuantResolvable, Dict[str, Any]]]
QuantSpecType = Union[QuantSpecElement, Tuple[QuantSpecElement, ...]]


def _grouped_mm_key(*args, **kwargs):
    raise RuntimeError('The canonical grouped-MM key must never be executed.')


def grouped_mm_functions() -> Tuple[Callable, ...]:
    """Return grouped-MM callables available in the current Torch/Transformers runtime."""
    functions = []
    for owner, name in ((torch, '_grouped_mm'), (torch.nn.functional, 'grouped_mm')):
        func = getattr(owner, name, None)
        if func is not None:
            functions.append(func)
    for namespace, name in (('aten', '_grouped_mm'), ('transformers', 'grouped_mm_fallback')):
        try:
            packet = getattr(getattr(torch.ops, namespace), name)
            # Accessing an unknown torch.ops attribute creates a placeholder packet.
            if packet._schemas:
                functions.append(packet)
                default = getattr(packet, 'default', None)
                if default is not None:
                    functions.append(default)
        except (AttributeError, RuntimeError):
            pass
    return tuple(dict.fromkeys(functions))


def _canonical_function(func: Callable) -> Callable:
    return _grouped_mm_key if any(
        func is candidate for candidate in grouped_mm_functions()) else func


_CONTAINER_NAME = '_functional_quantizers'
_STATE_NAME = '_functional_quantization_state'
_MISSING = object()
_PARAMETER_DISPATCH_FUNCTIONS = {
    _grouped_mm_key,
    torch.nn.functional.linear,
    torch.bmm,
    torch.matmul,
    torch.Tensor.matmul,
    torch.Tensor.__matmul__}
_FUNCTION_ARGUMENT_NAMES = {
    torch.nn.functional.linear: ('input', 'weight', 'bias'),
    torch.bmm: ('input', 'mat2'),
    torch.matmul: ('input', 'other'),
    torch.nn.functional.conv1d: ('input', 'weight', 'bias'),
    torch.nn.functional.conv2d: ('input', 'weight', 'bias'),
    torch.nn.functional.conv3d: ('input', 'weight', 'bias'),
    torch.nn.functional.conv_transpose1d: ('input', 'weight', 'bias'),
    torch.nn.functional.conv_transpose2d: ('input', 'weight', 'bias'),
    torch.nn.functional.conv_transpose3d: ('input', 'weight', 'bias'),}
_FUNCTION_ARGUMENT_NAMES[torch.Tensor.__matmul__] = ('input', 'other')
_FUNCTION_ARGUMENT_NAMES[torch.Tensor.matmul] = ('input', 'other')
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    _FUNCTION_ARGUMENT_NAMES[torch.nn.functional.scaled_dot_product_attention] = (
        'query', 'key', 'value')
_FUNCTION_ARGUMENT_NAMES[_grouped_mm_key] = (('input', 'self', 'mat_a'),
                                             ('weight', 'mat2', 'mat_b'),
                                             'offs',
                                             'bias',
                                             'out_dtype')


def _is_di_kwargs_pair(element: Any) -> bool:
    """Return whether an element is a ``(quantizer, di_kwargs)`` pair."""
    return isinstance(element, tuple) and len(element) == 2 and isinstance(element[1], dict)


def _parse_quant_map(quant_map: Dict[Callable, QuantSpecType]) -> Dict[Callable, List[Any]]:
    """Normalize each function specification to a positional list."""
    parsed = {}
    for func, spec in quant_map.items():
        func = _canonical_function(func)
        normalized = list(spec) if isinstance(spec, tuple) and not _is_di_kwargs_pair(spec) else [
            spec]
        if func in parsed and parsed[func] != normalized:
            raise ValueError('Grouped-MM aliases must use the same functional quantization spec.')
        parsed[func] = normalized
    return parsed


def _split_spec_element(element: Any) -> Tuple[QuantResolvable, Dict[str, Any]]:
    """Split a specification element into a quantizer and DI overrides."""
    if _is_di_kwargs_pair(element):
        return element[0], dict(element[1])
    return element, {}


def _resolve_spec(element: Any, module: nn.Module, module_name: str,
                  index: int) -> Tuple[Optional[Type], Dict[str, Any]]:
    """Resolve a class, ``None``, or resolver callable for one call site."""
    if element is _MISSING:
        return None, {}
    quantizer, di_kwargs = _split_spec_element(element)
    if quantizer is None or isinstance(quantizer, type):
        return quantizer, di_kwargs
    if not callable(quantizer):
        raise TypeError(f'Invalid functional quantizer spec {quantizer!r}.')
    resolved_quantizer, resolved_di_kwargs = _split_spec_element(
        quantizer(module, module_name, index))
    if resolved_quantizer is not None and not isinstance(resolved_quantizer, type):
        raise TypeError(
            'Functional quantizer resolvers must return a quantizer class, '
            '(quantizer class, DI kwargs), or None.')
    return resolved_quantizer, {**di_kwargs, **resolved_di_kwargs}


def _module_key(
        module_name: str,
        func: Callable,
        function_index: int,
        index: int,
        arg_idx: int,
        weight: bool = False) -> str:
    # ModuleDict names must not contain dots. The configured function ordinal makes
    # same-named callables distinct without process-specific object IDs.

    """Create a deterministic ``ModuleDict`` key for a call-site quantizer."""
    safe_module = module_name.replace('.', '__') or 'root'
    func_name = getattr(func, '__name__', 'function')
    suffix = f'_arg{arg_idx}_wq' if weight else ('' if arg_idx == 0 else f'_arg{arg_idx}')
    return f'_fq_{safe_module}_{func_name}_{function_index}_{index}{suffix}'


def _logical_arguments(
        func: Callable, args: Tuple[Any, ...],
        kwargs: Dict[str, Any]) -> Tuple[List[Tuple[int, Any, Callable[[Any], None]]], List[Any]]:
    """Return positional and known keyword tensor slots with write-back callbacks."""
    values = list(args)
    slots = []
    names = _FUNCTION_ARGUMENT_NAMES.get(func, ())
    for index, value in enumerate(values):
        slots.append(
            (index, value, lambda replacement, index=index: values.__setitem__(index, replacement)))
    for index in range(len(values), len(names)):
        aliases = names[index] if isinstance(names[index], tuple) else (names[index],)
        name = next((alias for alias in aliases if alias in kwargs), None)
        if name is not None:
            slots.append((
                index,
                kwargs[name],
                lambda replacement,
                name=name: kwargs.__setitem__(name, replacement)))
    return slots, values


class _WeightQuantHolder(nn.Module):
    """Adapter exposing explicitly configured operation metadata to weight solvers."""

    def __init__(self, weight: nn.Parameter, output_channel_dim: int) -> None:
        """Expose a weight and explicit output channel dimension to a proxy."""
        super().__init__()
        self.weight = weight
        self.bias = None
        self.output_channel_dim = output_channel_dim
        self.out_channels = weight.shape[output_channel_dim]


def _storage_tensor(value: Tensor) -> Tensor:
    """Return the tensor whose storage is indexed by a functional weight view."""
    if isinstance(value, QuantTensor):
        return value._value_ if getattr(value, '_is_groupwise', False) else value.value
    return value


class _QuantParametrization(nn.Module):

    def __init__(self, state: 'FunctionalQuantState', proxy: nn.Module, owner_id: str) -> None:
        """Store the mode state and proxy that quantize a parameter on demand."""
        super().__init__()
        self._state = state
        self.proxy = proxy
        self.owner_id = owner_id

    def forward(self, value: Tensor) -> Tensor:
        """Return the original parameter or its quantized proxy output."""
        if not self._state.enabled:
            return value
        if getattr(self.proxy, 'disable_quant', False):
            quantized_value = value
        else:
            quantized_value = self.proxy(value)
        if self._state.linear_observers:
            self._state._record_runtime_weight(self.owner_id, quantized_value)
        return quantized_value


@dataclass
class _PreparedArgument:
    quantizer_key: str


@dataclass
class _PreparedCall:
    arguments: Dict[int, _PreparedArgument] = field(default_factory=dict)


@dataclass
class _DiscoveredArgument:
    quant_class: Optional[Type]
    di_kwargs: Dict[str, Any]
    parameter_owner: Optional[Tuple[nn.Module, str]] = None
    fallback_quant_class: Optional[Type] = None
    fallback_di_kwargs: Dict[str, Any] = field(default_factory=dict)
    example_device: Optional[torch.device] = None
    view_indices: Tuple[int, ...] = ()
    transpose_weight: bool = False


@dataclass
class _OwnerPlan:
    quant_class: Type
    di_kwargs: Dict[str, Any]
    quantizer_key: str
    transpose_weight: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class _FunctionalWeightOwner:
    owner: nn.Module
    owner_name: str
    parameter_name: str
    proxy: nn.Module
    parametrization: _QuantParametrization

    @property
    def id(self) -> str:
        return f'{self.owner_name + ":" if self.owner_name else ""}{self.parameter_name}'

    @property
    def original_parameter(self) -> nn.Parameter:
        return getattr(self.owner.parametrizations, self.parameter_name).original


@dataclass(frozen=True)
class FunctionalLinearObservation:
    """An intercepted functional linear operation resolved to an owner view."""

    owner_id: str
    view_indices: Tuple[int, ...]
    input: Tensor
    function: Callable
    module: nn.Module
    module_name: str
    call_index: int


class FunctionalQuantState:
    """Prepared functional quantization state.

    Quantizer modules remain registered on the model after a mode exits. Call
    :meth:`cleanup` or :func:`remove_functional_quantization` to remove them.
    """

    def __init__(self, model: nn.Module, quant_map: Dict[Callable, QuantSpecType]) -> None:
        """Attach the retained quantizer container and initialize prepared state."""
        self.model = model
        self.quant_map = quant_map
        self.specs = _parse_quant_map(quant_map)
        self.function_indices = {func: index for index, func in enumerate(self.specs)}
        self.calls: Dict[Tuple[str, Callable, int], _PreparedCall] = {}
        self.owners: Dict[str, _FunctionalWeightOwner] = {}
        self.linear_views: Dict[Tuple[str, Tuple[int, ...]], bool] = {}
        self.linear_observers: List[Callable[[FunctionalLinearObservation], None]] = []
        self.runtime_weights: Dict[str, Tensor] = {}
        self.grouped_view_calls: Dict[Tuple[str, Callable, int], Tuple[str, Tuple[int, ...]]] = {}
        self.counter_resetters: List[Callable[[], None]] = []
        self.registered_parametrizations: List[Tuple[nn.Module, str]] = []
        self.parametrizations_removed = False
        self.enabled = False
        self._closed = False
        if hasattr(model, _CONTAINER_NAME):
            raise RuntimeError(
                'Model already has functional quantizers. Clean up its existing state first.')
        model.add_module(_CONTAINER_NAME, nn.ModuleDict())
        object.__setattr__(model, _STATE_NAME, self)

    @property
    def quantizers(self) -> nn.ModuleDict:
        """Return the model-owned registry of prepared quantizer modules."""
        return getattr(self.model, _CONTAINER_NAME)

    def remove_parametrizations(self) -> None:
        """Remove functional weight parametrizations and restore original parameters."""
        had_parametrizations = bool(self.registered_parametrizations)
        for owner, name in reversed(self.registered_parametrizations):
            if is_parametrized(owner, name):
                parametrizations = getattr(owner.parametrizations, name)
                if any(isinstance(item, _QuantParametrization) for item in parametrizations):
                    remove_parametrizations(owner, name, leave_parametrized=False)
        self.registered_parametrizations.clear()
        self.parametrizations_removed |= had_parametrizations

    def cleanup(self) -> None:
        """Remove all functional quantization mutations from the model."""
        if self._closed:
            return
        self.enabled = False
        self.remove_parametrizations()
        if hasattr(self.model, _CONTAINER_NAME):
            delattr(self.model, _CONTAINER_NAME)
        if getattr(self.model, _STATE_NAME, None) is self:
            delattr(self.model, _STATE_NAME)
        self.calls.clear()
        self.owners.clear()
        self.linear_views.clear()
        self.linear_observers.clear()
        self.runtime_weights.clear()
        self.grouped_view_calls.clear()
        self.counter_resetters.clear()
        self._closed = True

    def reset_active_counters(self) -> None:
        """Restart prepared call-site ordinals for a nested functional forward."""
        for reset in tuple(self.counter_resetters):
            reset()

    def register_linear_observer(self, observer: Callable[[FunctionalLinearObservation], None]):
        """Observe parameter-backed functional linear calls during an active mode."""
        self.linear_observers.append(observer)

        class _Handle:

            def remove(handle_self) -> None:
                if observer in self.linear_observers:
                    self.linear_observers.remove(observer)
                if not self.linear_observers:
                    self.runtime_weights.clear()

        return _Handle()

    def iter_linear_views(
            self,
            module_scope: Optional[nn.Module] = None) -> List[Tuple[str, Tuple[int, ...], bool]]:
        """Return owner IDs, indices, and layouts for prepared linear views."""
        views = [(owner_id, indices, transpose_weight) for (owner_id, indices),
                 transpose_weight in self.linear_views.items()]
        if module_scope is None:
            return views
        modules = set(module_scope.modules())
        return [view for view in views if self.owners[view[0]].owner in modules]

    def _record_runtime_weight(self, owner_id: str, value: Tensor) -> None:
        """Remember a root layout only while an observer needs expert identity."""
        self.runtime_weights[owner_id] = _storage_tensor(value)

    @staticmethod
    def _view_indices(root: Tensor, value: Tensor) -> Optional[Tuple[int, ...]]:
        """Recover a supported leading-index prefix from a storage-sharing view."""
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

    def _owner_view_from_weight(self, value: Any) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """Resolve an owner and index prefix from a runtime storage-sharing view."""
        if not isinstance(value, Tensor):
            return None
        value = _storage_tensor(value)
        roots = {owner_id: owner.original_parameter for owner_id, owner in self.owners.items()}
        roots.update(self.runtime_weights)
        for owner_id, root in roots.items():
            indices = self._view_indices(_storage_tensor(root), value)
            if indices is not None:
                return owner_id, indices
        return None

    def _linear_view_from_weight(self, value: Any) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """Resolve a runtime weight to a prepared owner-view key."""
        owner_view = self._owner_view_from_weight(value)
        if owner_view is None or owner_view not in self.linear_views:
            return None
        return owner_view

    def _assert_open(self) -> None:
        """Raise if this state was already cleaned up."""
        if self._closed:
            raise RuntimeError('Functional quantization state has been cleaned up.')
        if self.parametrizations_removed:
            raise RuntimeError(
                'Functional weight parametrizations have been removed; prepare a new state.')


class _HookedMode(TorchFunctionMode):

    def __init__(self, state: FunctionalQuantState) -> None:
        """Initialize interception state shared by preparation and application."""
        super().__init__()
        self.state = state
        self.model = state.model
        self.module_stack: List[Tuple[str, nn.Module]] = []
        self.counters = defaultdict(lambda: defaultdict(int))
        self.hooks: List[RemovableHandle] = []

    def _attach_hooks(self) -> None:
        """Attach hooks that maintain the active module stack and counters."""
        excluded = set(self.state.quantizers.modules())
        for name, module in self.model.named_modules():
            if module in excluded or isinstance(module, _QuantParametrization):
                continue
            pre_hook = self._pre_hook(name)
            post_hook = self._post_hook(name)
            pre_hook._brevitas_functional_quantization_hook = True
            post_hook._brevitas_functional_quantization_hook = True
            self.hooks.append(module.register_forward_pre_hook(pre_hook))
            self.hooks.append(module.register_forward_hook(post_hook, always_call=True))

        def reset_hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
            self._reset_hook(module, args, output)

        reset_hook._brevitas_functional_quantization_hook = True
        self.hooks.append(self.model.register_forward_hook(reset_hook, always_call=True))

    def _remove_hooks(self) -> None:
        """Remove managed hooks and discard transient forward state."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_stack.clear()
        self.counters.clear()

    def _pre_hook(self, name: str) -> Callable:
        """Create a pre-hook that resets and records each managed forward root."""

        def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
            if not self.module_stack:
                self.counters.clear()
            self.module_stack.append((name, module))

        return hook

    def _notify_linear_observers(
            self,
            name: str,
            module: nn.Module,
            func: Callable,
            index: int,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> None:
        """Report supported parameter-backed linear calls without using call order as identity."""
        if not self.state.linear_observers:
            return
        slots, _ = _logical_arguments(func, args, kwargs)
        values = {arg_idx: value for arg_idx, value, _ in slots}
        if func is _grouped_mm_key:
            grouped_view = self.state.grouped_view_calls.get((name, func, index))
            inp, weight, offsets = values.get(0), values.get(1), values.get(2)
            if grouped_view is None or not isinstance(inp, Tensor) or not isinstance(offsets,
                                                                                     Tensor):
                return
            owner_id, prefix = grouped_view
            runtime_owner_view = self.state._owner_view_from_weight(weight)
            if runtime_owner_view is not None and runtime_owner_view[0] == owner_id:
                prefix = runtime_owner_view[1]
            view_indices = sorted(
                indices for candidate_owner_id,
                indices,
                _ in self.state.iter_linear_views()
                if candidate_owner_id == owner_id and indices[:len(prefix)] == prefix)
            boundaries = [int(offset) for offset in offsets.detach().cpu().tolist()]
            if len(boundaries) != len(view_indices) or any(
                    end < start for start, end in zip((0, *boundaries), boundaries)) or (
                        boundaries and boundaries[-1] != inp.shape[0]):
                raise RuntimeError('Grouped-MM offsets do not match the prepared expert views.')
            start = 0
            for indices, end in zip(view_indices, boundaries):
                if end > start:
                    observation = FunctionalLinearObservation(
                        owner_id, indices, inp[start:end], func, module, name, index)
                    for observer in tuple(self.state.linear_observers):
                        observer(observation)
                start = end
            return
        if func not in (torch.nn.functional.linear,
                        torch.matmul,
                        torch.Tensor.matmul,
                        torch.Tensor.__matmul__):
            return
        if not isinstance(values.get(0), Tensor):
            return
        owner_view = self.state._linear_view_from_weight(values.get(1))
        if owner_view is None:
            return
        observation = FunctionalLinearObservation(
            owner_view[0], owner_view[1], values[0], func, module, name, index)
        for observer in tuple(self.state.linear_observers):
            observer(observation)

    def _post_hook(self, name: str) -> Callable:
        """Create an always-call hook that removes a completed module entry."""

        def hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
            """Perform this functional quantization operation."""
            if self.module_stack and self.module_stack[-1][0] == name:
                self.module_stack.pop()

        return hook

    def _reset_hook(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
        """Clear per-forward state after the top-level model invocation."""
        self.module_stack.clear()
        self.counters.clear()

    def _build_parameter_owners(self) -> None:
        """Map each direct model parameter to its owning module attribute."""
        self.parameter_owners.clear()
        self.aliased_parameters.clear()
        excluded = set(self.state.quantizers.modules())
        for _, module in self.model.named_modules():
            if module in excluded or isinstance(module, ParametrizationList):
                continue
            for name, parameter in module.named_parameters(recurse=False):
                owner = (module, name)
                if id(parameter
                     ) in self.parameter_owners and self.parameter_owners[id(parameter)] != owner:
                    self.aliased_parameters.add(id(parameter))
                self.parameter_owners[id(parameter)] = owner
            for name in getattr(module, 'parametrizations', {}):
                original = getattr(module.parametrizations, name).original
                owner = (module, name)
                if id(original
                     ) in self.parameter_owners and self.parameter_owners[id(original)] != owner:
                    self.aliased_parameters.add(id(original))
                self.parameter_owners[id(original)] = owner

    def _parameter_owner(self, value: Tensor) -> Tuple[Optional[Tuple[nn.Module, str]], bool]:
        """Resolve a tensor to its registered parameter owner.

        Direct parameters are matched by object identity. Tensor views are
        matched by following their ``_base`` chain, which lets functional
        quantization distinguish parameter-derived operands from ordinary
        runtime tensors before applying activation-quantizer fallback. If a
        parameter-like operand misses, the owner map is rebuilt to account for
        parameters rematerialized or replaced by offload hooks.

        Returns ``((owner_module, parameter_name), is_direct_parameter)`` on a
        match, or ``(None, False)`` when the tensor is unrelated to a parameter.
        """

        def lookup():
            owner = self.parameter_owners.get(id(value))
            if owner is not None:
                return owner, True
            base = getattr(value, '_base', None)
            visited = set()
            while base is not None and id(base) not in visited:
                visited.add(id(base))
                owner = self.parameter_owners.get(id(base))
                if owner is not None:
                    return owner, False
                base = getattr(base, '_base', None)
            return None, False

        owner, is_direct = lookup()
        if owner is not None:
            return owner, is_direct
        base = value
        visited = set()
        while getattr(base, '_base', None) is not None and id(base) not in visited:
            visited.add(id(base))
            base = base._base
        if not isinstance(base, nn.Parameter):
            return None, False
        # Offload hooks can replace or materialize a parameter after the initial map.
        self._build_parameter_owners()
        return lookup()

    def _spec_for(self, func: Callable, arg_idx: int, is_parameter: bool) -> Any:
        """Select the effective specification for an argument at a call site."""
        specs = self.state.specs[func]
        if func in _PARAMETER_DISPATCH_FUNCTIONS and len(specs) == 3:
            if arg_idx == 0:
                return specs[2] if is_parameter else specs[0]
            if arg_idx == 1:
                return specs[2] if is_parameter else specs[1]
            return _MISSING
        if arg_idx < len(specs):
            return specs[arg_idx]
        # The task requires a missing second runtime spec to reuse argument zero.
        if arg_idx == 1 and not is_parameter and specs:
            return specs[0]
        return _MISSING

    def _add_quantizer(self, key: str, quantizer: nn.Module) -> None:
        """Register a newly prepared quantizer under a collision-free key."""
        if key in self.state.quantizers:
            raise RuntimeError(f'Duplicate functional quantizer key {key}.')
        self.state.quantizers[key] = quantizer

    def _create_activation(
            self, quant_class: Type, di_kwargs: Dict[str, Any], device: torch.device) -> nn.Module:
        """Create an activation quantizer on the observed tensor device."""
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class
        quantizer = QuantIdentity(act_quant=quant_injector, return_quant_tensor=True)
        quantizer.train(self.model.training)
        return quantizer.to(device)

    def _create_weight(
            self, quant_class: Type, di_kwargs: Dict[str, Any], value: nn.Parameter) -> nn.Module:
        # Per-channel/groupwise operations must override this explicitly. A scalar
        # weight quantizer keeps the standard linear-layout default.

        """Create a weight proxy using explicit functional-operation metadata."""
        output_channel_dim = di_kwargs.get('output_channel_dim', 0)
        holder = _WeightQuantHolder(value, output_channel_dim)
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class.let()
        return quant_injector.proxy_class(holder, quant_injector).to(value.device)


class _FunctionalQuantBuilder(_HookedMode):

    def __init__(self, state: FunctionalQuantState) -> None:
        """Initialize a builder that discovers calls before parametrizing owners."""
        super().__init__(state)
        self.discovered_calls: Dict[Tuple[str, Callable, int], Dict[int, _DiscoveredArgument]] = {}
        self.owner_plans: Dict[Tuple[nn.Module, str], _OwnerPlan] = {}
        self.parameter_owners: Dict[int, Tuple[nn.Module, str]] = {}
        self.aliased_parameters = set()
        self.module_names: Dict[int, str] = {}

    def _build_parameter_owners(self) -> None:
        """Map parameters to owners and retain stable qualified names."""
        super()._build_parameter_owners()
        self.module_names = {id(module): name for name, module in self.model.named_modules()}

    @staticmethod
    def _view_is_transposed(owner: Tuple[nn.Module, str], value: Tensor) -> bool:
        owner_value = getattr(owner[0], owner[1])
        rank_delta = owner_value.dim() - value.dim()
        return rank_delta >= 0 and value.dim() >= 2 and tuple(value.shape) == (
            *owner_value.shape[rank_delta:-2], owner_value.shape[-1],
            owner_value.shape[-2]) and tuple(value.stride()) == (
                *owner_value.stride()[rank_delta:-2],
                owner_value.stride()[-1],
                owner_value.stride()[-2])

    @staticmethod
    def _view_indices(owner: Tuple[nn.Module, str], value: Tensor,
                      is_direct_parameter: bool) -> Optional[Tuple[int, ...]]:
        """Resolve direct or leading-index parameter views to stable owner indices."""
        if is_direct_parameter:
            return ()
        owner_value = getattr(owner[0], owner[1])
        rank_delta = owner_value.dim() - value.dim()
        is_leading_index = rank_delta >= 0 and tuple(value.shape) == tuple(
            owner_value.shape[rank_delta:]) and tuple(value.stride()) == tuple(
                owner_value.stride()[rank_delta:])
        if not is_leading_index and not _FunctionalQuantBuilder._view_is_transposed(owner, value):
            return None
        offset = value.storage_offset() - owner_value.storage_offset()
        indices = []
        for axis in range(rank_delta):
            stride = owner_value.stride()[axis]
            if stride == 0 or offset % stride:
                return None
            index = offset // stride
            if not 0 <= index < owner_value.shape[axis]:
                return None
            indices.append(index)
            offset -= index * stride
        return tuple(indices) if offset == 0 else None

    def _owner_quant_kwargs(
            self,
            value: Tensor,
            owner: Tuple[nn.Module, str],
            is_direct_parameter: bool,
            di_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Validate and normalize owner-level weight quantizer configuration."""
        owner_di_kwargs = dict(di_kwargs)
        owner_value = getattr(owner[0], owner[1], None)
        if not isinstance(owner_value, nn.Parameter):
            return owner_di_kwargs, 'its owner attribute is not an unparametrized Parameter'

        required = ('output_channel_dim', 'group_dim')
        if not is_direct_parameter:
            rank_delta = owner_value.dim() - value.dim()
            is_leading_index = rank_delta >= 0 and tuple(value.shape) == tuple(
                owner_value.shape[rank_delta:]) and tuple(value.stride()) == tuple(
                    owner_value.stride()[rank_delta:])
            is_last_two_transpose = self._view_is_transposed(owner, value)
            if not is_leading_index and not is_last_two_transpose:
                return owner_di_kwargs, (
                    'only leading-index views and final-two-axis transpose views are supported')
            missing = [name for name in required if name not in owner_di_kwargs]
            if missing:
                return owner_di_kwargs, (
                    'parameter-derived views require the weight quantizer to declare owner-level '
                    f"{', '.join(missing)}")

        axes = []
        owner_dim = owner_value.dim()
        for name in required:
            if name not in owner_di_kwargs:
                continue
            axis = owner_di_kwargs[name]
            if not isinstance(axis, int) or isinstance(axis,
                                                       bool) or not -owner_dim <= axis < owner_dim:
                return owner_di_kwargs, f'{name} is not a valid owner axis'
            axis = axis if axis >= 0 else owner_dim + axis
            owner_di_kwargs[name] = axis
            axes.append(axis)
        if len(axes) == 2 and axes[0] == axes[1]:
            return owner_di_kwargs, (
                'output_channel_dim and group_dim must refer to different owner axes')
        return owner_di_kwargs, None

    def _fallback_spec_for(self, func: Callable, arg_idx: int) -> Any:
        """Select an unambiguous runtime spec for failed owner quantization."""
        specs = self.state.specs[func]
        if func in _PARAMETER_DISPATCH_FUNCTIONS and len(specs) == 3 and arg_idx < 2:
            return specs[arg_idx]
        return _MISSING

    def _discover_argument(
            self, name: str, module: nn.Module, func: Callable, index: int, arg_idx: int,
            value: Tensor) -> Optional[_DiscoveredArgument]:
        """Classify one operand and record any owner-level weight requirement."""
        owner, is_direct_parameter = self._parameter_owner(value)
        spec = self._spec_for(func, arg_idx, owner is not None)
        quant_class, di_kwargs = _resolve_spec(spec, module, name, index)
        if quant_class is None:
            return None
        if owner is None:
            return _DiscoveredArgument(quant_class, di_kwargs)

        view_indices = self._view_indices(owner, value, is_direct_parameter)
        fallback_spec = self._fallback_spec_for(func, arg_idx)
        fallback_quant_class, fallback_di_kwargs = _resolve_spec(fallback_spec, module, name, index)
        owner_di_kwargs, error = self._owner_quant_kwargs(
            value, owner, is_direct_parameter, di_kwargs)
        owner_value = value if is_direct_parameter else getattr(owner[0], owner[1], None)
        if id(owner_value) in self.aliased_parameters:
            error = 'tied parameters do not have a unique owner attribute'
        if is_parametrized(owner[0], owner[1]):
            error = 'the owner is already parametrized'
        if view_indices is None:
            view_indices = ()
        if func is _grouped_mm_key and view_indices:
            error = 'indexed grouped-MM owner prefixes are unsupported'
        uses_rhs_matrix = func in (
            torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, _grouped_mm_key)
        transpose_weight = uses_rhs_matrix != self._view_is_transposed(owner, value)

        quantizer_key = _module_key(
            name, func, self.state.function_indices[func], index, arg_idx, weight=True)
        plan = self.owner_plans.get(owner)
        if plan is None:
            self.owner_plans[owner] = _OwnerPlan(
                quant_class=quant_class,
                di_kwargs=owner_di_kwargs,
                quantizer_key=quantizer_key,
                transpose_weight=transpose_weight,
                error=error)
        elif plan.error is None:
            if error is not None:
                plan.error = error
            elif (plan.quant_class is not quant_class or plan.di_kwargs != owner_di_kwargs or
                  plan.transpose_weight != transpose_weight):
                plan.error = 'the owner is used with incompatible quantizers or matrix layouts'
        return _DiscoveredArgument(
            quant_class,
            owner_di_kwargs,
            owner,
            fallback_quant_class,
            fallback_di_kwargs,
            value.device,
            view_indices,
            transpose_weight)

    def _discover_call(
            self,
            name: str,
            module: nn.Module,
            func: Callable,
            actual_func: Callable,
            index: int,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        """Record operand provenance and requirements without mutating the model."""
        slots, values = _logical_arguments(func, args, kwargs)
        call_key = (name, func, index)
        discovered = self.discovered_calls.setdefault(call_key, {})
        call = self.state.calls.setdefault(call_key, _PreparedCall())
        for arg_idx, value, replace in slots:
            if not isinstance(value, Tensor) or isinstance(value, QuantTensor):
                continue
            argument = self._discover_argument(name, module, func, index, arg_idx, value)
            if argument is not None:
                discovered[arg_idx] = argument
                if argument.parameter_owner is None:
                    key = _module_key(name, func, self.state.function_indices[func], index, arg_idx)
                    quantizer = self._create_activation(
                        argument.quant_class, argument.di_kwargs, value.device)
                    self._add_quantizer(key, quantizer)
                    call.arguments[arg_idx] = _PreparedArgument(key)
                    replace(quantizer(value))
        return actual_func(*tuple(values), **kwargs)

    def _register_owner_quantizers(self) -> None:
        """Finalize owner parametrizations and unsupported-view fallback mappings."""
        for owner, plan in self.owner_plans.items():
            owner_module, owner_name = owner
            if plan.error is not None:
                warnings.warn(
                    f"Parameter-derived operand '{owner_name}' on {type(owner_module).__name__} "
                    f"cannot be owner-quantized because {plan.error}; falling back to runtime "
                    "activation quantization when configured.",
                    UserWarning)
                continue
            parameter = getattr(owner_module, owner_name)
            proxy = self._create_weight(plan.quant_class, plan.di_kwargs, parameter)
            self._add_quantizer(plan.quantizer_key, proxy)
            owner_name_qualified = self.module_names[id(owner_module)]
            owner_id = f'{owner_name_qualified + ":" if owner_name_qualified else ""}{owner_name}'
            register_parametrization(
                owner_module, owner_name, _QuantParametrization(self.state, proxy, owner_id))
            self.state.registered_parametrizations.append(owner)
            owner_record = _FunctionalWeightOwner(
                owner_module,
                owner_name_qualified,
                owner_name,
                proxy,
                getattr(owner_module.parametrizations, owner_name)[-1])
            self.state.owners[owner_id] = owner_record

            # Preparation may observe only some routes through a stacked owner, so
            # enumerate every leading-index matrix with the same supported layout.
            original = owner_record.original_parameter
            direct_transpose_weight = False
            for call_key, argument_map in self.discovered_calls.items():
                for argument in argument_map.values():
                    # A batched operand does not identify one stable logical matrix view.
                    if argument.parameter_owner == owner and call_key[1] is torch.bmm:
                        continue
                    if argument.parameter_owner != owner:
                        continue
                    is_grouped_mm = call_key[1] is _grouped_mm_key
                    if is_grouped_mm:
                        self.state.grouped_view_calls[call_key] = (owner_id, argument.view_indices)
                    if not argument.view_indices and not is_grouped_mm:
                        if argument.view_indices == ():
                            direct_transpose_weight = argument.transpose_weight
                        continue
                    leading_dims = original.shape[:-2]
                    prefix = ()
                    for suffix in product(*(range(size) for size in leading_dims[len(prefix):])):
                        indices = (*prefix, *suffix)
                        key = (owner_id, tuple(indices))
                        existing = self.state.linear_views.get(key)
                        if existing is not None and existing != argument.transpose_weight:
                            raise RuntimeError(
                                f"Functional parameter '{owner_id}' is used with incompatible linear layouts."
                            )
                        self.state.linear_views[key] = argument.transpose_weight
            if original.dim() == 2:
                self.state.linear_views[(owner_id, ())] = direct_transpose_weight

        for call_key, arguments in self.discovered_calls.items():
            name, func, index = call_key
            call = self.state.calls[call_key]
            for arg_idx, argument in arguments.items():
                if argument.parameter_owner is None:
                    continue
                owner_plan = self.owner_plans[argument.parameter_owner]
                if owner_plan.error is not None and argument.fallback_quant_class is not None:
                    key = _module_key(name, func, self.state.function_indices[func], index, arg_idx)
                    quantizer = self._create_activation(
                        argument.fallback_quant_class,
                        argument.fallback_di_kwargs,
                        argument.example_device)
                    self._add_quantizer(key, quantizer)
                    call.arguments[arg_idx] = _PreparedArgument(key)

    def _run_forward(
            self, example_inputs: Optional[Tuple[Any, ...]],
            example_kwargs: Optional[Dict[str, Any]]) -> None:
        """Run one hooked preparation forward and reset transient counters."""
        self._attach_hooks()
        try:
            with self, torch.no_grad():
                self.model(*(example_inputs or ()), **(example_kwargs or {}))
        finally:
            self._remove_hooks()

    def build(
            self, example_inputs: Optional[Tuple[Any, ...]],
            example_kwargs: Optional[Dict[str, Any]]) -> FunctionalQuantState:
        """Discover calls once, then register owner and fallback quantizers."""
        self._build_parameter_owners()
        try:
            self._run_forward(example_inputs, example_kwargs)
            self._register_owner_quantizers()
        except Exception:
            self.state.cleanup()
            raise
        return self.state

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        """Create and apply quantizers while discovering each configured call."""
        kwargs = {} if kwargs is None else dict(kwargs)
        canonical_func = _canonical_function(func)
        if canonical_func not in self.state.specs or not self.module_stack:
            return func(*args, **kwargs)
        name, module = self.module_stack[-1]
        index = self.counters[name][canonical_func]
        self.counters[name][canonical_func] += 1
        self._notify_linear_observers(name, module, canonical_func, index, args, kwargs)
        return self._discover_call(name, module, canonical_func, func, index, args, kwargs)


class functional_quantization_mode(_HookedMode):
    """Apply a :class:`FunctionalQuantState` during a model forward/backward."""

    def __init__(
            self,
            state: FunctionalQuantState,
            enabled: bool = True,
            remove_parametrizations_on_exit: bool = False) -> None:
        """Configure application of a prepared state for one context lifetime."""
        state._assert_open()
        super().__init__(state)
        self.enabled = enabled
        self.remove_parametrizations_on_exit = remove_parametrizations_on_exit
        self._previous_enabled = False

    def __enter__(self) -> 'functional_quantization_mode':
        """Enable parametrizations, hooks, and torch-function interception."""
        self._attach_hooks()
        self._previous_enabled = self.state.enabled
        self.state.enabled = self.enabled
        self.state.counter_resetters.append(self.counters.clear)
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Restore mode state and remove hooks after the managed block exits."""
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.state.enabled = self._previous_enabled
            if self.counters.clear in self.state.counter_resetters:
                self.state.counter_resetters.remove(self.counters.clear)
            self._remove_hooks()
            if self.remove_parametrizations_on_exit:
                self.state.remove_parametrizations()

    def _unprepared_call_is_passthrough(
            self,
            func: Callable,
            module: nn.Module,
            name: str,
            index: int,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> bool:
        """Return whether an unseen call has no functional quantization enabled."""
        slots, _ = _logical_arguments(func, args, kwargs)
        for arg_idx, value, _ in slots:
            if not isinstance(value, Tensor) or isinstance(value, QuantTensor):
                continue
            runtime_spec = self._spec_for(func, arg_idx, False)
            runtime_quant, _ = _resolve_spec(runtime_spec, module, name, index)
            if runtime_quant is not None:
                return False
            base = value
            while getattr(base, '_base', None) is not None:
                base = base._base
            if isinstance(base, nn.Parameter):
                parameter_spec = self._spec_for(func, arg_idx, True)
                parameter_quant, _ = _resolve_spec(parameter_spec, module, name, index)
                if parameter_quant is not None:
                    return False
        return True

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        """Route an intercepted call through its prepared argument quantizers."""
        kwargs = {} if kwargs is None else dict(kwargs)
        canonical_func = _canonical_function(func)
        if not self.enabled or not self.state.enabled or canonical_func not in self.state.specs or not self.module_stack:
            return func(*args, **kwargs)
        name, module = self.module_stack[-1]
        index = self.counters[name][canonical_func]
        self.counters[name][canonical_func] += 1
        self._notify_linear_observers(name, module, canonical_func, index, args, kwargs)
        call = self.state.calls.get((name, canonical_func, index))
        if call is None:
            if self._unprepared_call_is_passthrough(canonical_func,
                                                    self.module_stack[-1][1],
                                                    name,
                                                    index,
                                                    args,
                                                    kwargs):
                return func(*args, **kwargs)
            raise RuntimeError(
                'No prepared quantizer found for this functional call site; ensure example inputs exercise it.'
            )
        slots, values = _logical_arguments(canonical_func, args, kwargs)
        for arg_idx, value, replace in slots:
            prepared = call.arguments.get(arg_idx)
            if prepared is None or isinstance(value, QuantTensor) or not isinstance(value, Tensor):
                continue
            replace(self.state.quantizers[prepared.quantizer_key](value))
        return func(*tuple(values), **kwargs)

    def checkpoint_context_fn(self) -> Callable[[], Tuple[Any, Any]]:
        """Return a non-reentrant checkpoint ``context_fn`` for recomputation."""
        if torch_version < version.parse('2.1'):
            raise RuntimeError(
                'Functional checkpointing requires PyTorch >= 2.1 and use_reentrant=False.')

        def context_fn() -> Tuple[Any, '_FunctionalQuantInterceptor']:
            """Return no-op forward and functional recompute contexts."""
            return contextlib.nullcontext(), _FunctionalQuantInterceptor(self)

        return context_fn


class _FunctionalQuantInterceptor(TorchFunctionMode):

    def __init__(self, parent: functional_quantization_mode) -> None:
        """Reuse a parent application's prepared state during recomputation."""
        super().__init__()
        self.parent = parent

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        """Delegate recompute interception to the active parent mode."""
        return functional_quantization_mode.__torch_function__(
            self.parent, func, types, args, kwargs)


def prepare_functional_quantization(
        model: nn.Module,
        quant_map: Dict[Callable, QuantSpecType],
        example_inputs: Optional[Tuple[Any, ...]] = None,
        example_kwargs: Optional[Dict[str, Any]] = None) -> FunctionalQuantState:
    """Discover functional call sites and instantiate their quantizers."""
    if example_inputs is None and example_kwargs is None:
        raise ValueError(
            'prepare_functional_quantization requires example_inputs and/or example_kwargs.')
    state = FunctionalQuantState(model, quant_map)
    return _FunctionalQuantBuilder(state).build(example_inputs, example_kwargs)


def remove_functional_quantization(model: nn.Module) -> None:
    """Remove functional quantization from *model* when its state is unavailable."""
    state = getattr(model, _STATE_NAME, None)
    if isinstance(state, FunctionalQuantState):
        state.cleanup()
        return
    if hasattr(model, _CONTAINER_NAME):
        delattr(model, _CONTAINER_NAME)
    for module in model.modules():
        for name in list(getattr(module, 'parametrizations', {}).keys()):
            items = getattr(module.parametrizations, name)
            if any(isinstance(item, _QuantParametrization) for item in items):
                remove_parametrizations(module, name, leave_parametrized=False)
