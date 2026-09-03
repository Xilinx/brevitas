# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
import contextlib
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
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
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.quant_tensor import QuantTensor

# Runtime quantization for calls to torch functional operators.

__all__ = [
    'DEFAULT_FUNCTIONAL_OPERATION_REGISTRY',
    'DEFAULT_FUNCTIONAL_QUANTIZER_FACTORY',
    'FunctionalInterceptor',
    'FunctionalOperation',
    'FunctionalOperationRegistry',
    'FunctionalQuantState',
    'FunctionalQuantizerFactory',
    'FunctionalWeightSource',
    'FunctionalWeightOwner',
    'functional_quantization_mode',
    'grouped_mm_functions',
    'prepare_functional_quantization',
    'remove_functional_quantization']

QuantResolverResult = Optional[Union[Type, Tuple[Optional[Type], Dict[str, Any]]]]
QuantResolver = Callable[[nn.Module, str, int], QuantResolverResult]
QuantResolvable = Optional[Union[Type, QuantResolver]]
QuantSpecElement = Union[QuantResolvable, Tuple[QuantResolvable, Dict[str, Any]]]
QuantSpecType = Union[QuantSpecElement, Tuple[QuantSpecElement, ...]]


class FunctionalQuantizerFactory(Protocol):
    """Create quantizer modules used by prepared functional operations.

    Returned modules are registered under the prepared model so device moves,
    training state, and state-dict handling follow normal ``nn.Module`` behavior.
    Weight factories must return a Brevitas-compatible weight proxy. Custom
    quantizers must not invoke operations present in the active functional map,
    since those calls would be attributed to the surrounding model call site.
    """

    def create_activation(
            self, model: nn.Module, quant_class: Type, di_kwargs: Dict[str, Any],
            device: torch.device) -> nn.Module:
        ...

    def create_weight(
            self, quant_class: Type, di_kwargs: Dict[str, Any],
            value: nn.Parameter) -> WeightQuantProxyFromInjectorBase:
        ...


class FunctionalWeightSource(Protocol):
    """Expose prepared functional weights to an external optimization mode.

    Consumers can enumerate owners within a module scope, suspend functional
    quantization while doing their own tensor work, and restart prepared call
    ordinals before a repeated logical forward. The protocol intentionally hides
    the concrete ``FunctionalQuantState`` implementation.
    """

    operation_registry: 'FunctionalOperationRegistry'

    def iter_weight_owners(self,
                           module_scope: Optional[nn.Module] = None
                          ) -> Sequence['FunctionalWeightOwner']:
        ...

    def suspend_quantization(self):
        ...

    def restart_call_sequence(self) -> None:
        ...


def _grouped_mm_key(*args, **kwargs):
    raise RuntimeError('The canonical grouped-MM key must never be executed.')


def grouped_mm_functions() -> Tuple[Callable, ...]:
    """Return grouped-MM aliases available in the current runtime.

    Torch and Transformers expose grouped matrix multiplication through different
    Python and dispatcher names across versions. Missing dispatcher packets are
    ignored and aliases are returned in discovery order without duplicates.
    """
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


@dataclass(frozen=True)
class FunctionalOperation:
    """Describe one functional operation and its logical arguments.

    ``canonical`` is the stable key used by prepared call plans, while ``aliases``
    are equivalent callables accepted at runtime. ``argument_names`` maps keyword
    spellings to logical positional arguments. When ``parameter_dispatch`` is
    enabled, a three-entry quantizer specification means
    ``(runtime_arg0_quant, runtime_arg1_quant, parameter_quant)`` for the first two
    logical arguments.
    """

    canonical: Callable
    aliases: Tuple[Callable, ...] = ()
    argument_names: Tuple[Any, ...] = ()
    parameter_dispatch: bool = False

    def argument(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], index: int) -> Any:
        """Return one logical argument from positional or registered keyword aliases."""
        if index < len(args):
            return args[index]
        if index >= len(self.argument_names):
            return None
        names = self.argument_names[index]
        aliases = names if isinstance(names, tuple) else (names,)
        return next((kwargs[name] for name in aliases if name in kwargs), None)


class FunctionalOperationRegistry:
    """Resolve functional callables to shared operation descriptions.

    Registries decouple interception from a fixed list of Torch functions and can
    be copied before adding project-specific operations or aliases. Re-registering
    a canonical callable replaces its metadata only in that registry instance.
    """

    def __init__(self) -> None:
        self._operations: Dict[Callable, FunctionalOperation] = {}
        self._aliases: Dict[Callable, FunctionalOperation] = {}

    def register(
            self,
            canonical: Callable,
            *,
            aliases: Tuple[Callable, ...] = (),
            argument_names: Tuple[Any, ...] = (),
            parameter_dispatch: bool = False) -> FunctionalOperation:
        """Register a canonical callable, its aliases, and logical argument names."""
        operation = FunctionalOperation(
            canonical=canonical,
            aliases=tuple(dict.fromkeys((canonical, *aliases))),
            argument_names=argument_names,
            parameter_dispatch=parameter_dispatch)
        previous = self._operations.get(canonical)
        for alias in operation.aliases:
            registered = self._aliases.get(alias)
            if registered is not None and registered is not previous:
                raise ValueError(f'Functional operation alias {alias!r} is already registered.')
        if previous is not None:
            for alias in previous.aliases:
                if self._aliases.get(alias) is previous:
                    del self._aliases[alias]
        self._operations[canonical] = operation
        for alias in operation.aliases:
            self._aliases[alias] = operation
        return operation

    def resolve(self, func: Callable) -> FunctionalOperation:
        """Return registered metadata or a positional-only description for ``func``."""
        operation = self._aliases.get(func)
        if operation is not None:
            return operation
        # Optional grouped-MM aliases can appear after this module is imported.
        grouped = self._operations.get(_grouped_mm_key)
        if grouped is not None and any(func is candidate for candidate in grouped_mm_functions()):
            return grouped
        return FunctionalOperation(canonical=func, aliases=(func,))

    def copy(self) -> 'FunctionalOperationRegistry':
        """Return an independent registry suitable for user customization."""
        registry = FunctionalOperationRegistry()
        for operation in self._operations.values():
            registry.register(
                operation.canonical,
                aliases=tuple(
                    alias for alias in operation.aliases if alias is not operation.canonical),
                argument_names=operation.argument_names,
                parameter_dispatch=operation.parameter_dispatch)
        return registry


def _default_functional_operation_registry() -> FunctionalOperationRegistry:
    registry = FunctionalOperationRegistry()
    registry.register(
        _grouped_mm_key,
        aliases=grouped_mm_functions(),
        argument_names=(('input', 'self', 'mat_a'), ('weight', 'mat2', 'mat_b'),
                        'offs',
                        'bias',
                        'out_dtype'),
        parameter_dispatch=True)
    registry.register(
        torch.nn.functional.linear,
        argument_names=('input', 'weight', 'bias'),
        parameter_dispatch=True)
    registry.register(torch.bmm, argument_names=('input', 'mat2'), parameter_dispatch=True)
    registry.register(torch.matmul, argument_names=('input', 'other'), parameter_dispatch=True)
    registry.register(
        torch.Tensor.matmul, argument_names=('input', 'other'), parameter_dispatch=True)
    registry.register(
        torch.Tensor.__matmul__, argument_names=('input', 'other'), parameter_dispatch=True)
    for func in (torch.nn.functional.conv1d,
                 torch.nn.functional.conv2d,
                 torch.nn.functional.conv3d,
                 torch.nn.functional.conv_transpose1d,
                 torch.nn.functional.conv_transpose2d,
                 torch.nn.functional.conv_transpose3d):
        registry.register(func, argument_names=('input', 'weight', 'bias'))
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        registry.register(
            torch.nn.functional.scaled_dot_product_attention,
            argument_names=('query', 'key', 'value'))
    return registry


DEFAULT_FUNCTIONAL_OPERATION_REGISTRY = _default_functional_operation_registry()


def _canonical_function(
    func: Callable,
    operation_registry: FunctionalOperationRegistry = DEFAULT_FUNCTIONAL_OPERATION_REGISTRY
) -> Callable:
    return operation_registry.resolve(func).canonical


_CONTAINER_NAME = '_functional_quantizers'
_STATE_NAME = '_functional_quantization_state'
_MISSING = object()


def _is_di_kwargs_pair(element: Any) -> bool:
    """Return whether an element is a ``(quantizer, di_kwargs)`` pair."""
    return isinstance(element, tuple) and len(element) == 2 and isinstance(element[1], dict)


def _parse_quant_map(
        quant_map: Dict[Callable, QuantSpecType],
        operation_registry: FunctionalOperationRegistry) -> Dict[Callable, List[Any]]:
    """Normalize each function specification to a positional list."""
    parsed = {}
    for func, spec in quant_map.items():
        func = _canonical_function(func, operation_registry)
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
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operation_registry: FunctionalOperationRegistry = DEFAULT_FUNCTIONAL_OPERATION_REGISTRY
) -> Tuple[List[Tuple[int, Any, Callable[[Any], None]]], List[Any]]:
    """Return positional and known keyword tensor slots with write-back callbacks."""
    values = list(args)
    slots = []
    names = operation_registry.resolve(func).argument_names
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


class _DefaultFunctionalQuantizerFactory:
    """Construct the standard Brevitas activation and weight quantizer modules.

    Activation classes are wrapped by ``QuantIdentity``. Weight classes create
    their configured proxy around a small holder that exposes parameter shape and
    output-channel metadata expected by Brevitas injectors.
    """

    def create_activation(
            self, model: nn.Module, quant_class: Type, di_kwargs: Dict[str, Any],
            device: torch.device) -> nn.Module:
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class
        quantizer = QuantIdentity(act_quant=quant_injector, return_quant_tensor=True)
        quantizer.train(model.training)
        return quantizer.to(device)

    def create_weight(
            self, quant_class: Type, di_kwargs: Dict[str, Any],
            value: nn.Parameter) -> WeightQuantProxyFromInjectorBase:
        output_channel_dim = di_kwargs.get('output_channel_dim', 0)
        holder = _WeightQuantHolder(value, output_channel_dim)
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class.let()
        return quant_injector.proxy_class(holder, quant_injector).to(value.device)


DEFAULT_FUNCTIONAL_QUANTIZER_FACTORY = _DefaultFunctionalQuantizerFactory()


class _QuantParametrization(nn.Module):
    """Quantize an owned parameter lazily while functional mode is enabled.

    Returning the original value when the state or proxy is disabled allows the
    same parametrized model to execute floating reference passes without removing
    and re-registering the parametrization.
    """

    def __init__(self, state: 'FunctionalQuantState', proxy: nn.Module) -> None:
        """Store the mode state and proxy that quantize a parameter on demand."""
        super().__init__()
        self._state = state
        self.proxy = proxy

    def forward(self, value: Tensor) -> Tensor:
        """Return the original parameter or its quantized proxy output."""
        if not self._state.enabled:
            return value
        if getattr(self.proxy, 'disable_quant', False):
            return value
        return self.proxy(value)


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
    operand_transposed: bool = False


@dataclass
class _OwnerPlan:
    quant_class: Type
    di_kwargs: Dict[str, Any]
    quantizer_key: str
    error: Optional[str] = None


@dataclass(frozen=True)
class FunctionalWeightOwner:
    """Describe one parameter owned by prepared functional quantization.

    The descriptor exposes the original writable parameter, its Brevitas weight
    proxy, and the parametrization that materializes the runtime weight. Each
    ``parameter_uses`` entry records ``(operation, argument index, transposed)``;
    consumers can use this generic metadata without depending on the concrete
    preparation state.
    """

    module: nn.Module
    module_name: str
    parameter_name: str
    proxy: WeightQuantProxyFromInjectorBase
    parametrization: nn.Module
    # (canonical function, logical argument index, final-two-axis transpose)
    parameter_uses: Tuple[Tuple[Callable, int, bool], ...] = ()

    @property
    def id(self) -> str:
        """Return the stable qualified identifier used by external consumers."""
        return f'{self.module_name + ":" if self.module_name else ""}{self.parameter_name}'

    @property
    def original_parameter(self) -> nn.Parameter:
        """Return the writable parameter stored behind the registered parametrization."""
        return getattr(self.module.parametrizations, self.parameter_name).original


class FunctionalQuantState:
    """Prepared functional quantization state.

    Preparation discovers functional call sites, registers quantizer modules on
    the model, and parametrizes supported parameter operands. Applying the state
    later requires the same enabled call sites and per-module ordinals for each
    canonical function used during the example forward. Quantizers and
    parametrizations remain registered after a
    mode exits; call :meth:`cleanup` or :func:`remove_functional_quantization` once
    the functional mode is no longer active.
    """

    def __init__(
        self,
        model: nn.Module,
        quant_map: Dict[Callable, QuantSpecType],
        operation_registry: FunctionalOperationRegistry = DEFAULT_FUNCTIONAL_OPERATION_REGISTRY,
        quantizer_factory: FunctionalQuantizerFactory = DEFAULT_FUNCTIONAL_QUANTIZER_FACTORY
    ) -> None:
        """Attach the retained quantizer container and initialize prepared state."""
        self.model = model
        self.quant_map = quant_map
        self.operation_registry = operation_registry
        self.quantizer_factory = quantizer_factory
        self.specs = _parse_quant_map(quant_map, operation_registry)
        self.function_indices = {func: index for index, func in enumerate(self.specs)}
        self.calls: Dict[Tuple[str, Callable, int], _PreparedCall] = {}
        self._weight_owners: Dict[str, FunctionalWeightOwner] = {}
        self._call_sequence_resetters: List[Callable[[], None]] = []
        self.registered_parametrizations: List[Tuple[nn.Module, str]] = []
        self.parametrizations_removed = False
        self.enabled = False
        self._mode_active = False
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
        """Remove functional weight parametrizations and restore original parameters.

        The prepared state exclusively owns these parametrization stacks until cleanup.
        """
        had_parametrizations = bool(self.registered_parametrizations)
        for owner, name in reversed(self.registered_parametrizations):
            if is_parametrized(owner, name):
                parametrizations = getattr(owner.parametrizations, name)
                if any(isinstance(item, _QuantParametrization) for item in parametrizations):
                    remove_parametrizations(owner, name, leave_parametrized=False)
        self.registered_parametrizations.clear()
        self.parametrizations_removed |= had_parametrizations

    def cleanup(self) -> None:
        """Remove all functional quantization mutations after its mode has exited."""
        if self._closed:
            return
        self.enabled = False
        self.remove_parametrizations()
        if hasattr(self.model, _CONTAINER_NAME):
            delattr(self.model, _CONTAINER_NAME)
        if getattr(self.model, _STATE_NAME, None) is self:
            delattr(self.model, _STATE_NAME)
        self.calls.clear()
        self._weight_owners.clear()
        self._call_sequence_resetters.clear()
        self._closed = True

    def restart_call_sequence(self) -> None:
        """Restart call-site ordinals before another logical forward.

        This is used when a consumer directly invokes the saved model ``forward``
        more than once inside one outer module call, such as paired quantized and
        floating reference passes.
        """
        for reset in tuple(self._call_sequence_resetters):
            reset()

    @contextlib.contextmanager
    def suspend_quantization(self):
        """Temporarily suspend all prepared functional quantization.

        Interception remains installed, but prepared activation quantizers and
        parameter proxies are bypassed. This lets an external optimizer perform
        tensor operations without recursively applying functional quantization.
        """
        previous_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous_enabled

    def iter_weight_owners(self,
                           module_scope: Optional[nn.Module] = None) -> List[FunctionalWeightOwner]:
        """Return prepared functional weight owners, optionally restricted to a subtree."""
        self._assert_open()
        owners = list(self._weight_owners.values())
        if module_scope is None:
            return owners
        modules = set(module_scope.modules())
        return [owner for owner in owners if owner.module in modules]

    def _assert_open(self) -> None:
        """Raise if this state was already cleaned up."""
        if self._closed:
            raise RuntimeError('Functional quantization state has been cleaned up.')
        if self.parametrizations_removed:
            raise RuntimeError(
                'Functional weight parametrizations have been removed; prepare a new state.')


class FunctionalInterceptor(TorchFunctionMode):
    """Base mode for composable functional interception.

    The mode resolves every Torch callable through a shared operation registry and
    delegates to :meth:`_intercept`. Subclasses own domain-specific hooks and
    transient state; the base owns dispatch, exception-safe hook cleanup, and
    scoped recursion suppression. A callback can be supplied for lightweight
    composition without defining another subclass.
    """

    def __init__(
            self,
            operation_registry: FunctionalOperationRegistry = DEFAULT_FUNCTIONAL_OPERATION_REGISTRY,
            callback: Optional[Callable] = None) -> None:
        super().__init__()
        self.operation_registry = operation_registry
        self.callback = callback
        self.hooks: List[RemovableHandle] = []
        self._suspend_depth = 0

    @contextlib.contextmanager
    def suspend(self):
        """Temporarily bypass this interceptor while preserving outer modes."""
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    @property
    def interception_suspended(self) -> bool:
        return self._suspend_depth > 0

    def _attach_hooks(self) -> None:
        """Attach domain-specific hooks, if any."""

    def _clear_interception_state(self) -> None:
        """Clear transient domain-specific state after hook removal."""

    def _remove_hooks(self) -> None:
        for hook in reversed(self.hooks):
            hook.remove()
        self.hooks.clear()
        self._clear_interception_state()

    def _intercept(
            self,
            operation: FunctionalOperation,
            func: Callable,
            types: Tuple[Type, ...],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        """Handle one resolved operation or delegate to its original callable."""
        if self.callback is not None:
            return self.callback(operation, func, types, args, kwargs)
        return func(*args, **kwargs)

    def __enter__(self) -> 'FunctionalInterceptor':
        try:
            self._attach_hooks()
            return super().__enter__()
        except Exception:
            self._remove_hooks()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._remove_hooks()

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        """Resolve ``func`` and dispatch it unless interception is suspended."""
        kwargs = {} if kwargs is None else dict(kwargs)
        if self.interception_suspended:
            return func(*args, **kwargs)
        operation = self.operation_registry.resolve(func)
        return self._intercept(operation, func, types, args, kwargs)


class _HookedMode(FunctionalInterceptor):

    def __init__(self, state: FunctionalQuantState) -> None:
        """Initialize interception state shared by preparation and application."""
        super().__init__(operation_registry=state.operation_registry)
        self.state = state
        self.model = state.model
        self.module_stack: List[Tuple[str, nn.Module]] = []
        self.counters = defaultdict(lambda: defaultdict(int))

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

    def _clear_interception_state(self) -> None:
        """Discard transient module and call-sequence state."""
        self.module_stack.clear()
        self.counters.clear()

    def _pre_hook(self, name: str) -> Callable:
        """Create a pre-hook that resets and records each managed forward root."""

        def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
            if not self.module_stack:
                self.counters.clear()
            self.module_stack.append((name, module))

        return hook

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
        if self.operation_registry.resolve(func).parameter_dispatch and len(specs) == 3:
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
        quantizer = self.state.quantizer_factory.create_activation(
            self.model, quant_class, di_kwargs, device)
        quantizer.train(self.model.training)
        return quantizer

    def _create_weight(
            self, quant_class: Type, di_kwargs: Dict[str, Any],
            value: nn.Parameter) -> WeightQuantProxyFromInjectorBase:
        # Per-channel/groupwise operations must override this explicitly. A scalar
        # weight quantizer keeps the standard linear-layout default.

        """Create a weight proxy using explicit functional-operation metadata."""
        quantizer = self.state.quantizer_factory.create_weight(quant_class, di_kwargs, value)
        quantizer.train(self.model.training)
        return quantizer


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
        if self.operation_registry.resolve(func).parameter_dispatch and len(
                specs) == 3 and arg_idx < 2:
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
        operand_transposed = self._view_is_transposed(owner, value)

        quantizer_key = _module_key(
            name, func, self.state.function_indices[func], index, arg_idx, weight=True)
        plan = self.owner_plans.get(owner)
        if plan is None:
            self.owner_plans[owner] = _OwnerPlan(
                quant_class=quant_class,
                di_kwargs=owner_di_kwargs,
                quantizer_key=quantizer_key,
                error=error)
        elif plan.error is None:
            if error is not None:
                plan.error = error
            elif plan.quant_class is not quant_class or plan.di_kwargs != owner_di_kwargs:
                plan.error = 'the owner is used with incompatible quantizers'
        return _DiscoveredArgument(
            quant_class,
            owner_di_kwargs,
            owner,
            fallback_quant_class,
            fallback_di_kwargs,
            value.device,
            view_indices,
            operand_transposed)

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
        slots, values = _logical_arguments(func, args, kwargs, self.operation_registry)
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
                owner_module, owner_name, _QuantParametrization(self.state, proxy))
            self.state.registered_parametrizations.append(owner)
            parameter_uses = []
            for call_key, argument_map in self.discovered_calls.items():
                for arg_idx, argument in argument_map.items():
                    if argument.parameter_owner != owner:
                        continue
                    parameter_uses.append((call_key[1], arg_idx, argument.operand_transposed))
            owner_record = FunctionalWeightOwner(
                owner_module,
                owner_name_qualified,
                owner_name,
                proxy,
                getattr(owner_module.parametrizations, owner_name)[-1],
                tuple(dict.fromkeys(parameter_uses)))
            self.state._weight_owners[owner_id] = owner_record

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
        with self, torch.no_grad():
            self.model(*(example_inputs or ()), **(example_kwargs or {}))

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

    def _intercept(
            self,
            operation: FunctionalOperation,
            func: Callable,
            types: Tuple[Type, ...],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        """Create and apply quantizers while discovering each configured call."""
        canonical_func = operation.canonical
        if canonical_func not in self.state.specs or not self.module_stack:
            return func(*args, **kwargs)
        name, module = self.module_stack[-1]
        index = self.counters[name][canonical_func]
        self.counters[name][canonical_func] += 1
        return self._discover_call(name, module, canonical_func, func, index, args, kwargs)


class functional_quantization_mode(_HookedMode):
    """Apply a prepared :class:`FunctionalQuantState` to model execution.

    The context enables functional weight parametrizations, tracks the currently
    executing module, and replaces arguments at prepared functional call sites.
    Enabled call sites not exercised during preparation raise at runtime. Only one
    context may use a state at a time; parametrizations persist until explicit
    state cleanup.
    """

    def __init__(self, state: FunctionalQuantState, enabled: bool = True) -> None:
        """Configure application of a prepared state for one context lifetime."""
        state._assert_open()
        super().__init__(state)
        self.enabled = enabled
        self._call_sequence_resetter = None

    def __enter__(self) -> 'functional_quantization_mode':
        """Enable parametrizations, hooks, and torch-function interception."""
        if self.state._mode_active:
            raise RuntimeError('Overlapping functional quantization contexts are unsupported.')
        self.state._mode_active = True
        self.state.enabled = self.enabled
        self._call_sequence_resetter = self.counters.clear
        self.state._call_sequence_resetters.append(self._call_sequence_resetter)
        try:
            return super().__enter__()
        except Exception:
            self.state._mode_active = False
            self.state.enabled = False
            if self._call_sequence_resetter in self.state._call_sequence_resetters:
                self.state._call_sequence_resetters.remove(self._call_sequence_resetter)
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Restore mode state and remove hooks after the managed block exits."""
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.state._mode_active = False
            self.state.enabled = False
            if self._call_sequence_resetter in self.state._call_sequence_resetters:
                self.state._call_sequence_resetters.remove(self._call_sequence_resetter)

    def _intercept(
            self,
            operation: FunctionalOperation,
            func: Callable,
            types: Tuple[Type, ...],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        """Route an intercepted call through its prepared argument quantizers."""
        canonical_func = operation.canonical
        if not self.enabled or not self.state.enabled or canonical_func not in self.state.specs or not self.module_stack:
            return func(*args, **kwargs)
        name, module = self.module_stack[-1]
        index = self.counters[name][canonical_func]
        self.counters[name][canonical_func] += 1
        call = self.state.calls.get((name, canonical_func, index))
        if call is None:
            raise RuntimeError(
                'No prepared quantizer found for this functional call site; ensure example inputs exercise it.'
            )
        slots, values = _logical_arguments(canonical_func, args, kwargs, self.operation_registry)
        for arg_idx, value, replace in slots:
            prepared = call.arguments.get(arg_idx)
            if prepared is None or isinstance(value, QuantTensor) or not isinstance(value, Tensor):
                continue
            replace(self.state.quantizers[prepared.quantizer_key](value))
        return func(*tuple(values), **kwargs)

    def checkpoint_context_fn(self) -> Callable[[], Tuple[Any, Any]]:
        """Return a checkpoint context that reapplies interception on recompute.

        Pass the returned callable to non-reentrant ``torch.utils.checkpoint`` while
        this mode remains active. The original forward uses the owning mode, while
        backward recomputation receives a lightweight interceptor that delegates
        back to it.
        """
        if torch_version < version.parse('2.1'):
            raise RuntimeError(
                'Functional checkpointing requires PyTorch >= 2.1 and use_reentrant=False.')

        def context_fn() -> Tuple[Any, FunctionalInterceptor]:
            """Return no-op forward and functional recompute contexts."""

            def recompute_callback(operation, func, types, args, kwargs):
                # A still-active parent mode must not process the delegated call twice.
                with self.suspend():
                    return self._intercept(operation, func, types, args, kwargs)

            return contextlib.nullcontext(), FunctionalInterceptor(
                operation_registry=self.operation_registry, callback=recompute_callback)

        return context_fn


def prepare_functional_quantization(
    model: nn.Module,
    quant_map: Dict[Callable, QuantSpecType],
    example_inputs: Optional[Tuple[Any, ...]] = None,
    example_kwargs: Optional[Dict[str, Any]] = None,
    *,
    operation_registry: FunctionalOperationRegistry = DEFAULT_FUNCTIONAL_OPERATION_REGISTRY,
    quantizer_factory: FunctionalQuantizerFactory = DEFAULT_FUNCTIONAL_QUANTIZER_FACTORY
) -> FunctionalQuantState:
    """Discover functional call sites and attach their quantizers to ``model``.

    Exactly one representative forward is executed from ``example_inputs`` and
    ``example_kwargs``. Every call site that will have quantization enabled later
    must be exercised by that forward. A map value can be one quantizer spec or a
    tuple of positional specs. Each spec accepts a quantizer class, ``None``, a
    ``(quantizer, dependency-injection kwargs)`` pair, or a resolver called as
    ``resolver(module, module_name, call_index)``. For parameter-dispatched binary
    operations, a three-entry tuple assigns runtime argument 0, runtime argument 1,
    and parameter quantization respectively. Otherwise a missing runtime argument
    1 specification reuses argument 0. ``operation_registry`` controls
    callable aliases and argument binding, while ``quantizer_factory`` constructs
    the registered activation and Brevitas-compatible weight modules.

    Returns:
        A prepared state that can be applied with
        :class:`functional_quantization_mode` and must eventually be cleaned up.
    """
    if example_inputs is None and example_kwargs is None:
        raise ValueError(
            'prepare_functional_quantization requires example_inputs and/or example_kwargs.')
    state = FunctionalQuantState(model, quant_map, operation_registry, quantizer_factory)
    return _FunctionalQuantBuilder(state).build(example_inputs, example_kwargs)


def remove_functional_quantization(model: nn.Module) -> None:
    """Clean up the functional quantization state attached to ``model``.

    The model is expected to have been prepared and no functional mode may still
    be active. This removes retained quantizers, parameter parametrizations, and
    the state attribute itself.
    """
    getattr(model, _STATE_NAME).cleanup()
