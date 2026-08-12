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
from typing import Tuple
from typing import Type
from typing import Union

from packaging import version
import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.parametrize import is_parametrized
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrize import remove_parametrizations
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

from brevitas import config
from brevitas import torch_version
from brevitas.core.scaling.standalone import ConstScaling
from brevitas.core.scaling.standalone import ParameterScaling
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.channel_splitting import GraphChannelSplitting
from brevitas.graph.equalize import EqualizeGraph
from brevitas.graph.fixed_point import CollapseConsecutiveConcats
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.graph.fixed_point import MoveSplitBatchNormBeforeCat
from brevitas.graph.quantize_impl import act_handler
from brevitas.graph.quantize_impl import add_output_quant_handler
from brevitas.graph.quantize_impl import inp_placeholder_handler
from brevitas.graph.quantize_impl import layer_handler
from brevitas.graph.quantize_impl import layerwise_layer_handler
from brevitas.graph.quantize_impl import residual_handler
from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph.standardize import RemoveStochasticModules
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.nn import QuantIdentity
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor

# Runtime quantization for calls to torch functional operators.

__all__ = [
    'FunctionalQuantState',
    'functional_quantization_mode',
    'prepare_functional_quantization',
    'remove_functional_quantization']

QuantResolver = Callable[[nn.Module, str, int], Optional[Type]]
QuantResolvable = Optional[Union[Type, QuantResolver]]
QuantSpecElement = Union[QuantResolvable, Tuple[QuantResolvable, Dict[str, Any]]]
QuantSpecType = Union[QuantSpecElement, Tuple[QuantSpecElement, ...]]

_CONTAINER_NAME = '_functional_quantizers'
_STATE_NAME = '_functional_quantization_state'
_MISSING = object()
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
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    _FUNCTION_ARGUMENT_NAMES[torch.nn.functional.scaled_dot_product_attention] = (
        'query', 'key', 'value')


def _is_di_kwargs_pair(element: Any) -> bool:
    return isinstance(element, tuple) and len(element) == 2 and isinstance(element[1], dict)


def _parse_quant_map(quant_map: Dict[Callable, QuantSpecType]) -> Dict[Callable, List[Any]]:
    parsed = {}
    for func, spec in quant_map.items():
        parsed[func] = list(spec) if isinstance(spec, tuple) and not _is_di_kwargs_pair(spec) else [
            spec]
    return parsed


def _split_spec_element(element: Any) -> Tuple[QuantResolvable, Dict[str, Any]]:
    if _is_di_kwargs_pair(element):
        return element[0], dict(element[1])
    return element, {}


def _resolve_spec(element: Any, module: nn.Module, module_name: str,
                  index: int) -> Tuple[Optional[Type], Dict[str, Any]]:
    if element is _MISSING:
        return None, {}
    quantizer, di_kwargs = _split_spec_element(element)
    if quantizer is None or isinstance(quantizer, type):
        return quantizer, di_kwargs
    if not callable(quantizer):
        raise TypeError(f'Invalid functional quantizer spec {quantizer!r}.')
    resolved = quantizer(module, module_name, index)
    if resolved is not None and not isinstance(resolved, type):
        raise TypeError('Functional quantizer resolvers must return a quantizer class or None.')
    return resolved, di_kwargs


def _call_key(module_name: str, func: Callable, index: int) -> Tuple[str, Callable, int]:
    return module_name, func, index


def _module_key(
        module_name: str,
        func: Callable,
        function_index: int,
        index: int,
        arg_idx: int,
        weight: bool = False) -> str:
    # ModuleDict names must not contain dots. The configured function ordinal makes
    # same-named callables distinct without process-specific object IDs.
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
        name = names[index]
        if name in kwargs:
            slots.append((
                index,
                kwargs[name],
                lambda replacement,
                name=name: kwargs.__setitem__(name, replacement)))
    return slots, values


class _WeightQuantHolder(nn.Module):
    """Adapter exposing explicitly configured operation metadata to weight solvers."""

    def __init__(self, weight: nn.Parameter, output_channel_dim: int) -> None:
        super().__init__()
        self.weight = weight
        self.bias = None
        self.output_channel_dim = output_channel_dim
        self.out_channels = weight.shape[output_channel_dim]


class _QuantParametrization(nn.Module):

    def __init__(self, state: 'FunctionalQuantState', proxy: nn.Module) -> None:
        super().__init__()
        self._state = state
        self.proxy = proxy

    def forward(self, value: Tensor) -> Tensor:
        if not self._state.enabled:
            return value
        return _unpack_quant_tensor(self.proxy(value))


@dataclass
class _PreparedArgument:
    quantizer_key: Optional[str]
    parameter_owner: Optional[Tuple[nn.Module, str]] = None


@dataclass
class _PreparedCall:
    arguments: Dict[int, _PreparedArgument] = field(default_factory=dict)


class FunctionalQuantState:
    """Prepared functional quantization state.

    Quantizer modules remain registered on the model after a mode exits. Call
    :meth:`cleanup` or :func:`remove_functional_quantization` to remove them.
    """

    def __init__(self, model: nn.Module, quant_map: Dict[Callable, QuantSpecType]) -> None:
        self.model = model
        self.quant_map = quant_map
        self.specs = _parse_quant_map(quant_map)
        self.function_indices = {func: index for index, func in enumerate(self.specs)}
        self.arg_quant_map = {
            func: [_split_spec_element(element)[0] for element in specs] for func,
            specs in self.specs.items()}
        self.arg_di_kwargs_map = {
            func: [_split_spec_element(element)[1] for element in specs] for func,
            specs in self.specs.items()}
        self.calls: Dict[Tuple[str, Callable, int], _PreparedCall] = {}
        self.parameter_owners: Dict[int, Tuple[nn.Module, str]] = {}
        self.aliased_parameters = set()
        self.registered_parametrizations: List[Tuple[nn.Module, str]] = []
        self.enabled = False
        self._closed = False
        if hasattr(model, _CONTAINER_NAME):
            raise RuntimeError(
                'Model already has functional quantizers. Clean up its existing state first.')
        model.add_module(_CONTAINER_NAME, nn.ModuleDict())
        object.__setattr__(model, _STATE_NAME, self)

    @property
    def quantizers(self) -> nn.ModuleDict:
        return getattr(self.model, _CONTAINER_NAME)

    def remove_parametrizations(self) -> None:
        """Remove functional weight parametrizations and restore original parameters."""
        for owner, name in reversed(self.registered_parametrizations):
            if is_parametrized(owner, name):
                parametrizations = getattr(owner.parametrizations, name)
                if any(isinstance(item, _QuantParametrization) for item in parametrizations):
                    remove_parametrizations(owner, name, leave_parametrized=False)
        self.registered_parametrizations.clear()

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
        self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError('Functional quantization state has been cleaned up.')


class _HookedMode(TorchFunctionMode):

    def __init__(self, state: FunctionalQuantState) -> None:
        super().__init__()
        self.state = state
        self.model = state.model
        self.module_stack: List[Tuple[str, nn.Module]] = []
        self.counters = defaultdict(lambda: defaultdict(int))
        self.hooks: List[RemovableHandle] = []
        # Compatibility aliases for callers that inspected the original mode.
        self._module_stack = self.module_stack
        self._counters = self.counters
        self._hook_handles = self.hooks

    def _attach_hooks(self) -> None:
        excluded = set(self.state.quantizers.modules())
        for name, module in self.model.named_modules():
            if module in excluded or isinstance(module, _QuantParametrization):
                continue
            self.hooks.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.hooks.append(module.register_forward_hook(self._post_hook(name), always_call=True))
        self.hooks.append(self.model.register_forward_hook(self._reset_hook, always_call=True))

    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_stack.clear()
        self.counters.clear()

    def _pre_hook(self, name: str) -> Callable:

        def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
            self.module_stack.append((name, module))

        return hook

    def _post_hook(self, name: str) -> Callable:

        def hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
            if self.module_stack and self.module_stack[-1][0] == name:
                self.module_stack.pop()

        return hook

    def _reset_hook(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
        self.module_stack.clear()
        self.counters.clear()

    def _build_parameter_owners(self) -> None:
        self.state.parameter_owners.clear()
        for _, module in self.model.named_modules():
            if module in set(self.state.quantizers.modules()):
                continue
            for name, parameter in module.named_parameters(recurse=False):
                if id(parameter) in self.state.parameter_owners:
                    self.state.aliased_parameters.add(id(parameter))
                self.state.parameter_owners[id(parameter)] = (module, name)

    def _spec_for(self, func: Callable, arg_idx: int, num_args: int, is_parameter: bool) -> Any:
        specs = self.state.specs[func]
        if arg_idx < len(specs):
            # Three specs retain the established binary runtime/weight convention.
            if num_args == 2 and len(specs) == 3 and arg_idx == 1:
                return specs[2] if is_parameter else specs[1]
            return specs[arg_idx]
        # The task requires a missing second runtime spec to reuse argument zero.
        if arg_idx == 1 and not is_parameter and specs:
            return specs[0]
        return _MISSING

    def _add_quantizer(self, key: str, quantizer: nn.Module) -> None:
        if key in self.state.quantizers:
            raise RuntimeError(f'Duplicate functional quantizer key {key}.')
        self.state.quantizers[key] = quantizer

    def _create_activation(
            self, quant_class: Type, di_kwargs: Dict[str, Any], value: Tensor) -> nn.Module:
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class
        quantizer = QuantIdentity(act_quant=quant_injector, return_quant_tensor=True)
        quantizer.train(self.model.training)
        return quantizer.to(value.device)

    def _create_weight(
            self, quant_class: Type, di_kwargs: Dict[str, Any], value: nn.Parameter) -> nn.Module:
        # Per-channel/groupwise operations must override this explicitly. A scalar
        # weight quantizer keeps the standard linear-layout default.
        output_channel_dim = di_kwargs.get('output_channel_dim', 0)
        holder = _WeightQuantHolder(value, output_channel_dim)
        quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class.let()
        return quant_injector.proxy_class(holder, quant_injector).to(value.device)


class _FunctionalQuantBuilder(_HookedMode):

    def build(
            self, example_inputs: Optional[Tuple[Any, ...]],
            example_kwargs: Optional[Dict[str, Any]]) -> FunctionalQuantState:
        self._build_parameter_owners()
        self._attach_hooks()
        self.state.enabled = True
        try:
            with self, torch.no_grad():
                self.model(*(example_inputs or ()), **(example_kwargs or {}))
        except Exception:
            self.state.cleanup()
            raise
        finally:
            self.state.enabled = False
            self._remove_hooks()
        return self.state

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        kwargs = {} if kwargs is None else dict(kwargs)
        if func not in self.state.quant_map or not self.module_stack:
            return func(*args, **kwargs)
        name, module = self.module_stack[-1]
        index = self.counters[name][func]
        self.counters[name][func] += 1
        call_key = _call_key(name, func, index)
        call = self.state.calls.setdefault(call_key, _PreparedCall())
        slots, values = _logical_arguments(func, args, kwargs)
        for arg_idx, value, replace in slots:
            if not isinstance(value, Tensor) or isinstance(value, QuantTensor):
                continue
            owner = self.state.parameter_owners.get(id(value))
            if owner is not None and id(value) in self.state.aliased_parameters:
                raise RuntimeError(
                    'Functional weight quantization does not support tied parameters.')
            spec = self._spec_for(func, arg_idx, len(slots), owner is not None)
            quant_class, di_kwargs = _resolve_spec(spec, module, name, index)
            if quant_class is None:
                continue
            key = _module_key(
                name, func, self.state.function_indices[func], index, arg_idx, owner is not None)
            if owner is None:
                quantizer = self._create_activation(quant_class, di_kwargs, value)
                self._add_quantizer(key, quantizer)
                call.arguments[arg_idx] = _PreparedArgument(key)
                replace(quantizer(value))
            else:
                if is_parametrized(owner[0], owner[1]):
                    raise RuntimeError(
                        'Functional weight quantization does not support pre-parametrized weights.')
                proxy = self._create_weight(quant_class, di_kwargs, value)
                self._add_quantizer(key, proxy)
                owner_module, owner_name = owner
                register_parametrization(
                    owner_module, owner_name, _QuantParametrization(self.state, proxy))
                self.state.registered_parametrizations.append(owner)
                call.arguments[arg_idx] = _PreparedArgument(key, owner)
                replace(proxy(value))
        return func(*tuple(values), **kwargs)


class functional_quantization_mode(_HookedMode):
    """Apply a :class:`FunctionalQuantState` during a model forward/backward."""

    def __init__(
            self,
            state: FunctionalQuantState,
            enabled: bool = True,
            remove_parametrizations_on_exit: bool = False) -> None:
        state._assert_open()
        super().__init__(state)
        self.enabled = enabled
        self.remove_parametrizations_on_exit = remove_parametrizations_on_exit
        self._previous_enabled = False

    def __enter__(self) -> 'functional_quantization_mode':
        self._attach_hooks()
        self._previous_enabled = self.state.enabled
        self.state.enabled = self.enabled
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.state.enabled = self._previous_enabled
            self._remove_hooks()
            if self.remove_parametrizations_on_exit:
                self.state.remove_parametrizations()

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
        kwargs = {} if kwargs is None else dict(kwargs)
        if not self.enabled or not self.state.enabled or func not in self.state.quant_map or not self.module_stack:
            return func(*args, **kwargs)
        name, _ = self.module_stack[-1]
        index = self.counters[name][func]
        self.counters[name][func] += 1
        call = self.state.calls.get(_call_key(name, func, index))
        if call is None:
            raise RuntimeError(
                'No prepared quantizer found for this functional call site; ensure example inputs exercise it.'
            )
        slots, values = _logical_arguments(func, args, kwargs)
        for arg_idx, value, replace in slots:
            prepared = call.arguments.get(arg_idx)
            if prepared is None or isinstance(value, QuantTensor) or not isinstance(value, Tensor):
                continue
            if prepared.parameter_owner is not None:
                continue
            replace(self.state.quantizers[prepared.quantizer_key](value))
        return func(*tuple(values), **kwargs)

    def checkpoint_context_fn(self) -> Callable[[], Tuple[Any, Any]]:
        """Return a non-reentrant checkpoint ``context_fn`` for recomputation."""
        if torch_version < version.parse('2.1'):
            raise RuntimeError(
                'Functional checkpointing requires PyTorch >= 2.1 and use_reentrant=False.')

        def context_fn() -> Tuple[Any, '_FunctionalQuantInterceptor']:
            return contextlib.nullcontext(), _FunctionalQuantInterceptor(self)

        return context_fn


class _FunctionalQuantInterceptor(TorchFunctionMode):

    def __init__(self, parent: functional_quantization_mode) -> None:
        super().__init__()
        self.parent = parent

    def __torch_function__(
            self, func: Callable, types: Tuple[Type, ...], args=(), kwargs=None) -> Any:
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


COMPUTE_LAYER_MAP = {
    nn.AvgPool2d:
        None,
    nn.MultiheadAttention: (
        qnn.QuantMultiheadAttention,
        {
            'in_proj_weight_quant': Int8WeightPerTensorFloat,
            'in_proj_bias_quant': Int32Bias,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'v_quant': Int8ActPerTensorFloat,
            'out_proj_input_quant': Int8ActPerTensorFloat,
            'out_proj_weight_quant': Int8WeightPerTensorFloat,
            'out_proj_bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.RNN: (
        qnn.QuantRNN,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': True}),
    nn.LSTM: (
        qnn.QuantLSTM,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'sigmoid_quant': Uint8ActPerTensorFloat,
            'tanh_quant': Int8ActPerTensorFloat,
            'cell_state_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': True}),
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Conv3d: (
        qnn.QuantConv3d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose3d: (
        qnn.QuantConvTranspose3d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True})}

LayerMapValueType = Optional[Tuple[Type[torch.nn.Module], Dict[str, Any]]]

LAYERWISE_COMPUTE_LAYER_MAP: Dict[Type[torch.nn.Module], LayerMapValueType] = {
    nn.AvgPool2d:
        None,
    nn.MultiheadAttention: (
        qnn.QuantMultiheadAttention,
        {
            'in_proj_input_quant': Int8ActPerTensorFloat,
            'in_proj_weight_quant': Int8WeightPerTensorFloat,
            'in_proj_bias_quant': Int32Bias,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'v_quant': Int8ActPerTensorFloat,
            'out_proj_input_quant': Int8ActPerTensorFloat,
            'out_proj_weight_quant': Int8WeightPerTensorFloat,
            'out_proj_bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.LSTM: (
        qnn.QuantLSTM,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'sigmoid_quant': Uint8ActPerTensorFloat,
            'tanh_quant': Int8ActPerTensorFloat,
            'cell_state_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': False}),
    nn.RNN: (
        qnn.QuantRNN,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': False}),
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.Conv3d: (
        qnn.QuantConv3d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.ConvTranspose3d: (
        qnn.QuantConvTranspose3d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False})}

UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)

QUANT_ACT_MAP = {
    nn.ReLU: (qnn.QuantReLU, {
        'act_quant': Uint8ActPerTensorFloat, 'return_quant_tensor': True}),
    nn.ReLU6: (
        qnn.QuantReLU, {
            'act_quant': Uint8ActPerTensorFloatMaxInit, 'max_val': 6.,
            'return_quant_tensor': True}),
    nn.Hardtanh: (
        qnn.QuantHardTanh,
        {
            'act_quant': Int8ActPerTensorFloatMinMaxInit,
            'max_val': lambda module: module.max_val,
            'min_val': lambda module: module.min_val,
            'return_quant_tensor': True}),
    nn.Sigmoid:
        (qnn.QuantSigmoid, {
            'act_quant': Uint8ActPerTensorFloat,
            'return_quant_tensor': True,}),}

QUANT_IDENTITY_MAP = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': Int8ActPerTensorFloat, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': Uint8ActPerTensorFloat, 'return_quant_tensor': True}),}


def align_input_quant(
        module, shared_quant_identity, shared_quant_identity_name, quant_identity_map, align_sign):
    """
    Based on the input module, the function decides how to align its output.
    """
    # If it is a QuantIdentity already, simply modify tensor_quant or the scaling implementations
    # based on whether we need to align the sign or not
    if isinstance(module, qnn.QuantIdentity):
        if align_sign or module.input_quant.is_signed == shared_quant_identity.input_quant.is_signed:
            return shared_quant_identity
        else:
            assert not module.input_quant.is_signed and shared_quant_identity.input_quant.is_signed
            quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
            return (
                quant_module_class,
                {
                    **quant_module_kwargs,
                    'scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .scaling_impl,
                    'int_scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .int_scaling_impl})
    elif hasattr(module, 'output_quant'):
        return (type(module), {'output_quant': shared_quant_identity})
    # If it is a QuantAct where the scaling can be determined through stats (thus through calibration),
    # then adapt its act_quant according to align_sign.
    elif hasattr(module, 'act_quant') and not isinstance(
            module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
        (ParameterScaling, ConstScaling)):
        module_type = type(module)
        if align_sign:
            partial_config = {
                'signed':
                    shared_quant_identity.act_quant.is_signed,
                'tensor_quant':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant}
        else:
            partial_config = {
                'scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .scaling_impl,
                'int_scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .int_scaling_impl}
        injector = module.act_quant.quant_injector.let(**partial_config)
        return module_type(act_quant=injector, return_quant_tensor=True)
    # In all other cases, return the name of the QuantIdentity that will be added at the output of
    # the module
    else:
        return shared_quant_identity_name


def preprocess_for_quantize(
        model,
        trace_model=True,
        relu6_to_relu=True,
        equalize_iters=0,
        equalize_merge_bias=True,
        merge_bn=True,
        equalize_bias_shrinkage: str = 'vaiq',
        equalize_scale_computation: str = 'maxabs',
        channel_splitting_ratio: float = 0.0,
        channel_splitting_split_input: bool = True,
        channel_splitting_criterion: str = 'maxabs'):

    training_state = model.training
    model.eval()

    if trace_model:
        model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    if relu6_to_relu:
        model = ModuleToModuleByClass(nn.ReLU6, nn.ReLU).apply(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = CollapseConsecutiveConcats().apply(model)
    model = MoveSplitBatchNormBeforeCat().apply(model)
    if merge_bn:
        model = MergeBatchNorm().apply(model)
    model = RemoveStochasticModules().apply(model)
    model = EqualizeGraph(
        iterations=equalize_iters,
        merge_bias=equalize_merge_bias,
        bias_shrinkage=equalize_bias_shrinkage,
        scale_computation_type=equalize_scale_computation).apply(model)
    if channel_splitting_ratio > 0:
        model = GraphChannelSplitting(
            split_ratio=channel_splitting_ratio,
            split_criterion=channel_splitting_criterion,
            split_input=channel_splitting_split_input).apply(model)
    model.train(training_state)
    return model


def quantize(
        graph_model,
        quant_identity_map=QUANT_IDENTITY_MAP,
        compute_layer_map=COMPUTE_LAYER_MAP,
        quant_act_map=QUANT_ACT_MAP,
        unsigned_act_tuple=UNSIGNED_ACT_TUPLE,
        requantize_layer_handler_output=True):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    graph_model = inp_placeholder_handler(
        graph_model, input_quantizer=quant_identity_map.get('signed', None))
    graph_model = act_handler(graph_model, layer_map=quant_act_map)
    graph_model = add_output_quant_handler(
        graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple)
    # The call to esidual_handler has to be performed before layer_handler
    # so that all requantization steps are correctly inserted and aligned.
    graph_model = residual_handler(
        graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant)
    graph_model = layer_handler(
        graph_model,
        layer_map=compute_layer_map,
        quant_identity_map=quant_identity_map,
        quant_act_map=quant_act_map,
        unsigned_act_tuple=unsigned_act_tuple,
        requantize_output=requantize_layer_handler_output)
    graph_model = DisableLastReturnQuantTensor().apply(graph_model)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model


def layerwise_quantize(
        model: nn.Module,
        compute_layer_map: dict = LAYERWISE_COMPUTE_LAYER_MAP,
        name_blacklist=None):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = model.training
    model.eval()
    model = layerwise_layer_handler(
        model, layer_map=compute_layer_map, name_blacklist=name_blacklist)
    model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return model
