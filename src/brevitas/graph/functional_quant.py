# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepared activation quantization for torch functional calls."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

from brevitas.nn import QuantIdentity
from brevitas.quant_tensor import QuantTensor

__all__ = [
    'FunctionalQuantState',
    'functional_quantization_mode',
    'prepare_functional_quantization',
    'remove_functional_quantization']

QuantResolver = Callable[[nn.Module, str, int], Optional[Type]]
QuantSpec = Union[Type, QuantResolver, None, Tuple[Union[Type, QuantResolver, None], Dict[str, Any]]]

_CONTAINER_NAME = '_functional_quantizers'
_STATE_NAME = '_functional_quantization_state'
_MISSING = object()
_ARGUMENT_NAMES = {
    torch.nn.functional.linear: ('input', 'weight', 'bias'),
    torch.bmm: ('input', 'mat2'),
    torch.matmul: ('input', 'other')}
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    _ARGUMENT_NAMES[torch.nn.functional.scaled_dot_product_attention] = ('query', 'key', 'value')


def _key(module_name: str, func: Callable, index: int) -> str:
    """Create a deterministic key for a prepared quantizer."""
    safe_module = module_name.replace('.', '__') or 'root'
    return f'_fq_{safe_module}_{getattr(func, "__name__", "function")}_{index}'


@dataclass
class _PreparedCall:
    quantizer_keys: Dict[int, str]


class FunctionalQuantState:
    """Prepared activation quantizers for a model's functional call sites."""

    def __init__(self, model: nn.Module, quant_map: Dict[Callable, Any]) -> None:
        """Attach the retained quantizer container and initialize prepared state."""
        self.model = model
        self.quant_map = quant_map
        self.calls: Dict[Tuple[str, Callable, int], _PreparedCall] = {}
        self.closed = False
        if hasattr(model, _CONTAINER_NAME):
            raise RuntimeError('Model already has functional quantizers.')
        model.add_module(_CONTAINER_NAME, nn.ModuleDict())
        object.__setattr__(model, _STATE_NAME, self)

    @property
    def quantizers(self) -> nn.ModuleDict:
        """Return the model-owned registry of prepared quantizer modules."""
        return getattr(self.model, _CONTAINER_NAME)

    def cleanup(self) -> None:
        """Remove retained functional quantizers from the model."""
        if self.closed:
            return
        if hasattr(self.model, _CONTAINER_NAME):
            delattr(self.model, _CONTAINER_NAME)
        if getattr(self.model, _STATE_NAME, None) is self:
            delattr(self.model, _STATE_NAME)
        self.calls.clear()
        self.closed = True


class _FunctionalQuantMode(TorchFunctionMode):

    def __init__(self, state: FunctionalQuantState, build: bool = False, enabled: bool = True) -> None:
        """Initialize this functional quantization component."""
        super().__init__()
        self.state = state
        self.build = build
        self.enabled = enabled
        self.module_stack: List[Tuple[str, nn.Module]] = []
        self.counters = defaultdict(lambda: defaultdict(int))
        self.hooks: List[RemovableHandle] = []

    def _attach_hooks(self) -> None:
        """Attach hooks that track the active module and call count."""
        excluded = set(self.state.quantizers.modules())
        for name, module in self.state.model.named_modules():
            if module in excluded:
                continue
            self.hooks.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.hooks.append(module.register_forward_hook(self._post_hook(name), always_call=True))
        self.hooks.append(self.state.model.register_forward_hook(self._reset, always_call=True))

    def _remove_hooks(self) -> None:
        """Remove managed hooks and clear transient forward state."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.module_stack.clear()
        self.counters.clear()

    def _pre_hook(self, name: str) -> Callable:
        """Create a hook that records entry into a module."""
        def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
            """Perform this functional quantization operation."""
            self.module_stack.append((name, module))
        return hook

    def _post_hook(self, name: str) -> Callable:
        """Create a hook that removes a completed module entry."""
        def hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
            """Perform this functional quantization operation."""
            if self.module_stack and self.module_stack[-1][0] == name:
                self.module_stack.pop()
        return hook

    def _reset(self, module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
        """Reset module-stack and counter state after a model forward."""
        self.module_stack.clear()
        self.counters.clear()

    def _specs(self, func: Callable) -> List[Any]:
        """Normalize the configured specification for an intercepted function."""
        spec = self.state.quant_map[func]
        if isinstance(spec, tuple) and not (len(spec) == 2 and isinstance(spec[1], dict)):
            return list(spec)
        return [spec]

    def _resolve(self, spec: Any, module: nn.Module, name: str, index: int):
        """Resolve one quantizer specification for a call site."""
        if spec is _MISSING:
            return None, {}
        if isinstance(spec, tuple):
            spec, kwargs = spec
        else:
            kwargs = {}
        if spec is None or isinstance(spec, type):
            return spec, kwargs
        result = spec(module, name, index)
        if result is not None and not isinstance(result, type):
            raise TypeError('Functional quantizer resolvers must return a class or None.')
        return result, kwargs

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Handle an intercepted functional operation."""
        kwargs = {} if kwargs is None else kwargs
        if not self.enabled or func not in self.state.quant_map or not self.module_stack:
            return func(*args, **kwargs)
        module_name, module = self.module_stack[-1]
        index = self.counters[module_name][func]
        self.counters[module_name][func] += 1
        call_key = (module_name, func, index)
        values = list(args)
        names = _ARGUMENT_NAMES.get(func, ())
        slots = [(idx, value, lambda replacement, idx=idx: values.__setitem__(idx, replacement))
                 for idx, value in enumerate(values)]
        slots += [(idx, kwargs[name], lambda replacement, name=name: kwargs.__setitem__(name, replacement))
                  for idx, name in enumerate(names[len(values):], len(values)) if name in kwargs]
        if self.build:
            call = _PreparedCall({})
            self.state.calls[call_key] = call
        else:
            call = self.state.calls.get(call_key)
            if call is None:
                raise RuntimeError('No prepared quantizer found for this functional call site.')
        specs = self._specs(func)
        for arg_idx, value, replace in slots:
            if not isinstance(value, torch.Tensor) or isinstance(value, QuantTensor):
                continue
            spec = specs[arg_idx] if arg_idx < len(specs) else (specs[0] if arg_idx == 1 and specs else _MISSING)
            quant_class, di_kwargs = self._resolve(spec, module, module_name, index)
            if quant_class is None:
                continue
            if self.build:
                quantizer_key = _key(module_name, func, index) + ('' if arg_idx == 0 else f'_arg{arg_idx}')
                quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class
                quantizer = QuantIdentity(act_quant=quant_injector, return_quant_tensor=True).to(value.device)
                self.state.quantizers[quantizer_key] = quantizer
                call.quantizer_keys[arg_idx] = quantizer_key
            else:
                quantizer = self.state.quantizers[call.quantizer_keys[arg_idx]]
            replace(quantizer(value))
        return func(*tuple(values), **kwargs)


def prepare_functional_quantization(
        model: nn.Module,
        quant_map: Dict[Callable, Type],
        example_inputs: Optional[Tuple[Any, ...]] = None,
        example_kwargs: Optional[Dict[str, Any]] = None) -> FunctionalQuantState:
    """Run one example forward and create activation quantizers for its call sites."""
    if example_inputs is None and example_kwargs is None:
        raise ValueError('example_inputs and/or example_kwargs are required.')
    state = FunctionalQuantState(model, quant_map)
    mode = _FunctionalQuantMode(state, build=True)
    mode._attach_hooks()
    try:
        with mode, torch.no_grad():
            model(*(example_inputs or ()), **(example_kwargs or {}))
    except Exception:
        state.cleanup()
        raise
    finally:
        mode._remove_hooks()
    return state


class functional_quantization_mode(_FunctionalQuantMode):
    """Apply activation quantizers created by ``prepare_functional_quantization``."""

    def __init__(self, state: FunctionalQuantState, enabled: bool = True) -> None:
        """Configure application of a prepared state for one context lifetime."""
        if state.closed:
            raise RuntimeError('Functional quantization state has been cleaned up.')
        super().__init__(state, enabled=enabled)

    def __enter__(self):
        """Enable parametrizations, hooks, and torch-function interception."""
        self._attach_hooks()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore mode state and remove hooks after the managed block exits."""
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._remove_hooks()


def remove_functional_quantization(model: nn.Module) -> None:
    """Remove retained functional quantization state from ``model``."""
    state = getattr(model, _STATE_NAME, None)
    if isinstance(state, FunctionalQuantState):
        state.cleanup()
    elif hasattr(model, _CONTAINER_NAME):
        delattr(model, _CONTAINER_NAME)
