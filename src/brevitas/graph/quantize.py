# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

from packaging import version
import torch
from torch import nn
from torch import Tensor
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
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

if torch_version >= version.parse('1.12'):
    from collections import defaultdict

    from torch.nn.utils.parametrize import is_parametrized
    from torch.nn.utils.parametrize import register_parametrization
    from torch.nn.utils.parametrize import remove_parametrizations
    from torch.overrides import TorchFunctionMode

    from brevitas.nn import QuantIdentity
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import QuantTensor

    # A lambda/callable that resolves a quantizer class at runtime from the current
    # module instance, its module name, and the per-(module, func) call index.
    QuantResolver = Callable[[nn.Module, str, int], Optional[Type]]
    # A quantizer spec element is either a quantizer class, a resolver callable, or
    # None. Optionally it may be paired with a dict of dependency-injection kwargs
    # (e.g. ``group_dim``, ``output_channel_dim``) as ``(quantizer, di_kwargs)``.
    QuantResolvable = Optional[Union[Type, QuantResolver]]
    QuantSpecElement = Union[QuantResolvable, Tuple[QuantResolvable, Dict[str, Any]]]
    QuantSpecType = Union[QuantSpecElement, Tuple[QuantSpecElement, ...]]

    # ------------------------------------------------------------------------- #
    # Shared, phase-agnostic helpers used by both the preparation phase (which
    # creates quantizers/parametrizations) and the application phase (which only
    # routes intercepted calls to the pre-created quantizers).
    # ------------------------------------------------------------------------- #


    def _is_di_kwargs_pair(element: Any) -> bool:
        """Whether ``element`` is a ``(quantizer, di_kwargs)`` pair.

        Only a 2-tuple whose second item is a dict qualifies; quantizer classes,
        resolvers, and None are never dicts, so this is unambiguous."""
        return isinstance(element, tuple) and len(element) == 2 and isinstance(element[1], dict)

    def _split_spec_element(
            element: 'QuantSpecElement') -> Tuple['QuantResolvable', Dict[str, Any]]:
        """Split a spec element into a ``(quantizer, di_kwargs)`` pair.

        A spec element is either a bare quantizer/resolver/None, or a
        ``(quantizer, di_kwargs)`` tuple. Bare elements get an empty di_kwargs
        dict."""
        if _is_di_kwargs_pair(element):
            return element[0], dict(element[1])
        return element, {}

    def _parse_quant_map(
        quant_map: Dict[Callable, 'QuantSpecType']
    ) -> Tuple[Dict[Callable, List['QuantResolvable']], Dict[Callable, List[Dict[str, Any]]]]:
        """Parse a ``quant_map`` into per-arg quantizer and di_kwargs lists.

        Returns ``(arg_quant_map, arg_di_kwargs_map)`` where both are keyed by the
        intercepted function and hold one entry per positional argument slot."""
        arg_quant_map: Dict[Callable, List['QuantResolvable']] = {}
        arg_di_kwargs_map: Dict[Callable, List[Dict[str, Any]]] = {}
        for func, spec in quant_map.items():
            # A spec is positional only if it is a tuple that is not itself a single
            # ``(quantizer, di_kwargs)`` pair. This lets a lone spec element carry
            # di_kwargs without being mistaken for a per-arg tuple.
            if isinstance(spec, tuple) and not _is_di_kwargs_pair(spec):
                spec_elements = list(spec)
            else:
                spec_elements = [spec]
            quantizers: List['QuantResolvable'] = []
            di_kwargs: List[Dict[str, Any]] = []
            for element in spec_elements:
                quantizer, kwargs = _split_spec_element(element)
                quantizers.append(quantizer)
                di_kwargs.append(kwargs)
            arg_quant_map[func] = quantizers
            arg_di_kwargs_map[func] = di_kwargs
        return arg_quant_map, arg_di_kwargs_map

    def _make_quantizer_key(module_name: str, func: Callable, index: int, suffix: str = '') -> str:
        """Create a unique key for a quantizer instance.

        Dots in module_name are replaced with underscores because
        ``nn.Module.add_module`` does not allow dots in names."""
        func_name = getattr(func, '__name__', str(func))
        safe_name = module_name.replace('.', '_')
        return f'_fq_{safe_name}_{func_name}_{index}{suffix}'

    def _resolve_spec_element(
            elem: 'QuantResolvable', module: nn.Module, module_name: str,
            index: int) -> Optional[Type]:
        """Resolve a spec element to a concrete quantizer class.

        A spec element is either ``None``, a quantizer class, or a resolver
        callable. Resolver callables are invoked as ``elem(module, module_name,
        index)`` and must return a quantizer class (or ``None`` to skip
        quantization of the argument). Quantizer classes are returned unchanged."""
        if elem is None:
            return None
        # Quantizer classes are types; resolver lambdas are not.
        if isinstance(elem, type):
            return elem
        if callable(elem):
            return elem(module, module_name, index)
        return elem

    def _move_to_model_device(model: nn.Module, module: nn.Module) -> nn.Module:
        """Move a module to the same device as ``model``."""
        try:
            device = next(model.parameters()).device
            module = module.to(device)
        except StopIteration:
            pass
        return module

    class _WeightQuantHolder(nn.Module):
        """Dummy module that holds a weight parameter so that
        WeightQuantProxyFromInjector can be instantiated through the standard
        QuantProxyMixin path.

        Only the weight tensor is exposed: any op-specific dependency-injection
        attributes (e.g. ``output_channel_dim``, ``group_dim``) must be provided
        explicitly on the quantizer via the ``quant_map`` spec."""

        def __init__(self, weight: nn.Parameter) -> None:
            super().__init__()
            self.weight = weight
            self.bias = None

    class _QuantParametrization(nn.Module):
        """Parametrization module that quantizes a parameter (e.g., weight) on-the-fly
        during forward using a weight quantization proxy."""

        def __init__(self, weight_quant_proxy: nn.Module) -> None:
            super().__init__()
            self.weight_quant_proxy = weight_quant_proxy

        def forward(self, x: Tensor) -> Tensor:
            out = self.weight_quant_proxy(x)
            return _unpack_quant_tensor(out)

    class FunctionalQuantState:
        """State produced by :func:`prepare_functional_quantization` and consumed by
        :class:`functional_quantization_mode`.

        It holds everything the application phase needs to route intercepted torch
        function calls to the quantizers created during preparation:

        - ``model``: the model that was prepared.
        - ``quant_map`` / ``arg_quant_map`` / ``arg_di_kwargs_map``: the parsed spec.
        - ``quantizers``: registry of created quantizers, keyed by call site
          (``{key: nn.Module}``), also registered as ``_fq_``-prefixed submodules on
          the model.
        - ``param_to_module``: maps a parameter ``data_ptr`` to ``(module, name)``.
        - ``parametrized_slots``: ``(module_name, func, arg_idx)`` slots that are
          handled by a persistent weight parametrization.
        - ``registered_parametrizations``: ``(module, param_name)`` entries this
          state registered, so they can be torn down via
          :meth:`remove_parametrizations`.

        Weight parametrizations registered during preparation are persistent and
        self-applying: they quantize the owning module's weight on every forward,
        independently of whether a :class:`functional_quantization_mode` block is
        active. They live until :meth:`remove_parametrizations` is called (either
        explicitly, or automatically by a
        ``functional_quantization_mode(..., remove_parametrizations_on_exit=True)``
        block on exit)."""

        def __init__(
                self,
                model: nn.Module,
                quant_map: Dict[Callable, 'QuantSpecType'],
                arg_quant_map: Dict[Callable, List['QuantResolvable']],
                arg_di_kwargs_map: Dict[Callable, List[Dict[str, Any]]],
                quantizers: Dict[str, nn.Module],
                param_to_module: Dict[int, Tuple[nn.Module, str]],
                parametrized_slots: Set[Tuple[str, Callable, int]],
                registered_parametrizations: List[Tuple[nn.Module, str]]) -> None:
            self.model = model
            self.quant_map = quant_map
            self.arg_quant_map = arg_quant_map
            self.arg_di_kwargs_map = arg_di_kwargs_map
            self.quantizers = quantizers
            self.param_to_module = param_to_module
            self.parametrized_slots = parametrized_slots
            self.registered_parametrizations = registered_parametrizations

        def remove_parametrizations(self) -> None:
            """Remove all weight parametrizations registered during preparation.

            Leaves the (now quantized) weights in place. Safe to call multiple
            times; subsequent calls are no-ops."""
            for owner_module, param_name in self.registered_parametrizations:
                if is_parametrized(owner_module, param_name):
                    remove_parametrizations(owner_module, param_name, leave_parametrized=True)
            self.registered_parametrizations.clear()
            self.parametrized_slots.clear()

    class _FunctionalQuantHooks:
        """Mixin providing the module-stack + per-module call-counter hooks shared by
        the preparation and application phases.

        Both phases must identify the innermost ``nn.Module`` executing a given
        intercepted call, plus the per-(module, func) call index, to compute a
        stable quantizer key. This mixin attaches forward pre/post hooks to every
        module of ``self.model`` to maintain that state.

        Expects the host class to define ``self.model``, ``self._module_stack``,
        ``self._counters`` and ``self._hook_handles``."""

        def _pre_hook(self, module_name: str) -> Callable:
            """Create a forward pre-hook that pushes the module onto the stack.

            The per-module call counter is reset every time the module is entered.
            This keeps the call ``index`` a function of how many times ``func`` is
            called within this specific module invocation only, which makes the
            original forward and a gradient-checkpointing recompute of the same
            region produce identical indices (and therefore reuse the same
            quantizers/parametrizations)."""

            def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
                self._module_stack.append((module_name, module))
                self._counters[module_name].clear()

            return hook

        def _post_hook(self, module_name: str) -> Callable:
            """Create a forward hook that pops the module from the stack."""

            def hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
                if self._module_stack and self._module_stack[-1][0] == module_name:
                    self._module_stack.pop()

            return hook

        def _reset_counters(self) -> None:
            """Reset all per-module call counters."""
            self._counters.clear()

        def _attach_hooks(self) -> None:
            """Attach pre/post forward hooks to all modules of the model."""
            for name, module in self.model.named_modules():
                # Skip quantizer modules that were added by preparation.
                if name.startswith('_fq_'):
                    continue
                pre_handle = module.register_forward_pre_hook(self._pre_hook(name))
                post_handle = module.register_forward_hook(self._post_hook(name))
                self._hook_handles.append(pre_handle)
                self._hook_handles.append(post_handle)

            # Reset counters after each complete top-level forward.
            def reset_hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
                self._reset_counters()

            handle = self.model.register_forward_hook(reset_hook)
            self._hook_handles.append(handle)

        def _remove_hooks(self) -> None:
            """Remove all attached hooks."""
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

    class _FunctionalQuantBuilder(_FunctionalQuantHooks, TorchFunctionMode):
        """Preparation-phase interceptor.

        Runs a single example forward with ``TorchFunctionMode`` + module-stack
        hooks active, and for every intercepted call it lazily creates the required
        activation quantizers (registered as ``_fq_``-prefixed submodules) and
        registers persistent weight parametrizations. The populated
        :class:`FunctionalQuantState` is returned by :func:`build`.

        The builder applies quantizers during its own forward as well, but that
        output is discarded: calibration and training happen later under
        :class:`functional_quantization_mode`."""

        def __init__(self, model: nn.Module, quant_map: Dict[Callable, 'QuantSpecType']) -> None:
            super().__init__()
            self.model = model
            self.quant_map = quant_map
            self._module_stack: List[Tuple[str, nn.Module]] = []
            self._counters: Dict[str, Dict[Callable, int]] = defaultdict(lambda: defaultdict(int))
            self._quantizers: Dict[str, nn.Module] = {}
            self._hook_handles: List[RemovableHandle] = []
            self._param_to_module: Dict[int, Tuple[nn.Module, str]] = {}
            self._registered_parametrizations: List[Tuple[nn.Module, str]] = []
            self._parametrized_slots: Set[Tuple[str, Callable, int]] = set()
            self._arg_quant_map, self._arg_di_kwargs_map = _parse_quant_map(quant_map)

        def build(
                self,
                example_inputs: Optional[Tuple[Any, ...]],
                example_kwargs: Optional[Dict[str, Any]]) -> FunctionalQuantState:
            """Run the example forward and return the populated state."""
            args = () if example_inputs is None else tuple(example_inputs)
            kwargs = {} if example_kwargs is None else dict(example_kwargs)
            self._build_param_to_module_map()
            self._attach_hooks()
            try:
                with self, torch.no_grad():
                    self.model(*args, **kwargs)
            finally:
                self._remove_hooks()
            return FunctionalQuantState(
                model=self.model,
                quant_map=self.quant_map,
                arg_quant_map=self._arg_quant_map,
                arg_di_kwargs_map=self._arg_di_kwargs_map,
                quantizers=self._quantizers,
                param_to_module=self._param_to_module,
                parametrized_slots=self._parametrized_slots,
                registered_parametrizations=self._registered_parametrizations)

        def _build_param_to_module_map(self) -> None:
            """Build a mapping from parameter data_ptr to (module, param_name)."""
            self._param_to_module.clear()
            for name, module in self.model.named_modules():
                if name.startswith('_fq_'):
                    continue
                for param_name, param in module.named_parameters(recurse=False):
                    self._param_to_module[param.data_ptr()] = (module, param_name)

        def _create_act_quantizer(
                self, quant_class: Type, di_kwargs: Dict[str, Any]) -> QuantIdentity:
            """Create a QuantIdentity quantizer for activation tensors.

            Any dependency-injection overrides (e.g. ``group_dim``) are applied to
            the quantizer via ``let`` before instantiation."""
            act_quant = quant_class.let(**di_kwargs) if di_kwargs else quant_class
            quantizer = QuantIdentity(act_quant=act_quant, return_quant_tensor=True)
            quantizer.train(self.model.training)
            return _move_to_model_device(self.model, quantizer)

        def _create_weight_quant_proxy(
                self, quant_class: Type, weight_param: nn.Parameter,
                di_kwargs: Dict[str, Any]) -> nn.Module:
            """Create a weight quantization proxy through a weight-holder module.

            Any dependency-injection overrides (e.g. ``output_channel_dim``,
            ``group_dim``) are applied to the quantizer via ``let``. Attributes that
            can be derived from the weight tensor (e.g. ``out_channels``,
            ``weight_ndims``) are resolved by the solver from the holder's weight."""
            holder = _WeightQuantHolder(weight_param)
            quant_injector = quant_class.let(**di_kwargs) if di_kwargs else quant_class.let()
            proxy = quant_injector.proxy_class(holder, quant_injector)
            return _move_to_model_device(self.model, proxy)

        def _get_or_create_act_quantizer(
                self,
                module_name: str,
                func: Callable,
                index: int,
                arg_idx: int,
                quant_class: Type,
                di_kwargs: Dict[str, Any]) -> QuantIdentity:
            """Get an existing activation quantizer or create a new one."""
            suffix = '' if arg_idx == 0 else f'_arg{arg_idx}'
            key = _make_quantizer_key(module_name, func, index, suffix=suffix)
            if key not in self._quantizers:
                quantizer = self._create_act_quantizer(quant_class, di_kwargs)
                self._quantizers[key] = quantizer
                if not hasattr(self.model, key):
                    self.model.add_module(key, quantizer)
            return self._quantizers[key]

        def _register_weight_parametrization(
                self,
                param_tensor: nn.Parameter,
                func: Callable,
                module_name: str,
                index: int,
                arg_idx: int,
                quant_class: Type,
                di_kwargs: Dict[str, Any]) -> None:
            """Register a persistent quantization parametrization on the parameter-owning module."""
            data_ptr = param_tensor.data_ptr()
            if data_ptr not in self._param_to_module:
                return
            owner_module, param_name = self._param_to_module[data_ptr]
            # Check if this parametrization was already registered by us.
            if is_parametrized(owner_module, param_name):
                for p in getattr(owner_module.parametrizations, param_name):
                    if isinstance(p, _QuantParametrization):
                        return  # Already registered

            weight_quant_proxy = self._create_weight_quant_proxy(
                quant_class, param_tensor, di_kwargs)

            suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
            key = _make_quantizer_key(module_name, func, index, suffix=suffix)
            self._quantizers[key] = weight_quant_proxy
            if not hasattr(self.model, key):
                self.model.add_module(key, weight_quant_proxy)

            param_module = _QuantParametrization(weight_quant_proxy)
            register_parametrization(owner_module, param_name, param_module)
            self._registered_parametrizations.append((owner_module, param_name))

        def _fallback_spec_index(self, func: Callable) -> Optional[int]:
            """Return the index of the first non-None spec element for ``func``."""
            for idx, spec_element in enumerate(self._arg_quant_map.get(func, [])):
                if spec_element is not None:
                    return idx
            return None

        def _di_kwargs_at(self, func: Callable, spec_idx: Optional[int]) -> Dict[str, Any]:
            """Return the dependency-injection kwargs for a resolved spec index."""
            di_kwargs = self._arg_di_kwargs_map.get(func, [])
            if spec_idx is None or spec_idx >= len(di_kwargs):
                return {}
            return di_kwargs[spec_idx]

        def _effective_spec_index(
                self, func: Callable, arg_idx: int, num_args: int, is_param: bool) -> Optional[int]:
            """Select the spec-list index to use for a positional argument."""
            spec_elements = self._arg_quant_map.get(func, [])
            if arg_idx >= len(spec_elements):
                return None
            if num_args == 2 and len(spec_elements) == 3 and arg_idx == 1:
                return 2 if is_param else 1
            return arg_idx

        def _quantize_arg(
                self,
                args: List[Any],
                arg_idx: int,
                func: Callable,
                current_module: nn.Module,
                current_module_name: str,
                index: int) -> None:
            """Create (and apply) the quantizer for a single positional argument."""
            if arg_idx >= len(args):
                return
            arg = args[arg_idx]
            if isinstance(arg, QuantTensor):
                return
            if not isinstance(arg, torch.Tensor):
                return

            is_param = isinstance(arg, nn.Parameter) or arg.data_ptr() in self._param_to_module
            spec_idx = self._effective_spec_index(func, arg_idx, len(args), is_param)
            spec_element = None if spec_idx is None else self._arg_quant_map[func][spec_idx]
            quant_class = _resolve_spec_element(
                spec_element, current_module, current_module_name, index)
            di_kwargs = self._di_kwargs_at(func, spec_idx)

            if arg_idx == 0:
                if quant_class is not None:
                    quantizer = self._get_or_create_act_quantizer(
                        current_module_name, func, index, arg_idx, quant_class, di_kwargs)
                    args[arg_idx] = quantizer(arg)
            else:
                param_slot = (current_module_name, func, arg_idx)
                if param_slot in self._parametrized_slots:
                    # The parametrization has already quantized this tensor.
                    pass
                elif is_param and quant_class is not None:
                    # Register a persistent parametrization on the owning module. On
                    # this first call the parametrization is not active yet, so we
                    # also quantize explicitly; on subsequent forwards the
                    # parametrization handles it automatically.
                    wq_suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
                    wq_key = _make_quantizer_key(current_module_name, func, index, suffix=wq_suffix)
                    self._register_weight_parametrization(
                        arg, func, current_module_name, index, arg_idx, quant_class, di_kwargs)
                    self._parametrized_slots.add(param_slot)
                    weight_quant_proxy = self._quantizers[wq_key]
                    args[arg_idx] = weight_quant_proxy(arg)
                elif not is_param:
                    # Regular tensor: use the explicit quantizer class if provided,
                    # otherwise fall back to the first non-None spec element.
                    effective_class = quant_class
                    effective_di_kwargs = di_kwargs
                    if effective_class is None:
                        fallback_idx = self._fallback_spec_index(func)
                        fallback_element = (
                            None
                            if fallback_idx is None else self._arg_quant_map[func][fallback_idx])
                        effective_class = _resolve_spec_element(
                            fallback_element, current_module, current_module_name, index)
                        effective_di_kwargs = self._di_kwargs_at(func, fallback_idx)
                    if effective_class is not None:
                        quantizer = self._get_or_create_act_quantizer(
                            current_module_name,
                            func,
                            index,
                            arg_idx,
                            effective_class,
                            effective_di_kwargs)
                        args[arg_idx] = quantizer(arg)

        def __torch_function__(
                self,
                func: Callable,
                types: Tuple[Type, ...],
                args: Tuple[Any, ...] = (),
                kwargs: Optional[Dict[str, Any]] = None) -> Any:
            if kwargs is None:
                kwargs = {}

            if func not in self.quant_map or not self._module_stack:
                return func(*args, **kwargs)

            current_module_name, current_module = self._module_stack[-1]
            index = self._counters[current_module_name][func]
            self._counters[current_module_name][func] += 1

            args = list(args)
            num_quant_args = len(self._arg_quant_map.get(func, []))
            for arg_idx in range(min(num_quant_args, len(args))):
                self._quantize_arg(args, arg_idx, func, current_module, current_module_name, index)
            return func(*tuple(args), **kwargs)

    def prepare_functional_quantization(
            model: nn.Module,
            quant_map: Dict[Callable, 'QuantSpecType'],
            example_inputs: Optional[Tuple[Any, ...]] = None,
            example_kwargs: Optional[Dict[str, Any]] = None) -> FunctionalQuantState:
        """Prepare a model for functional quantization.

        Runs a single example forward to discover every intercepted call site, and
        for each site creates the required activation quantizers (registered as
        ``_fq_``-prefixed submodules on ``model``) and registers persistent weight
        parametrizations. Returns a :class:`FunctionalQuantState` to be passed to
        :class:`functional_quantization_mode` for the application phase.

        Args:
            model: The nn.Module to prepare. Its forward is executed once.
            quant_map: Mapping from torch functions (e.g.
                ``torch.nn.functional.linear``) to a quantizer spec. See
                :class:`functional_quantization_mode` for the supported spec shapes.
            example_inputs: Positional args passed as ``model(*example_inputs)``.
            example_kwargs: Keyword args passed as ``model(**example_kwargs)``.

        At least one of ``example_inputs`` / ``example_kwargs`` must be provided so
        that the model can be run to discover the call sites to quantize.

        Note:
            Weight parametrizations registered here are persistent and self-applying;
            they remain on the model until :meth:`FunctionalQuantState.remove_parametrizations`
            is called (either explicitly, or by a
            ``functional_quantization_mode(..., remove_parametrizations_on_exit=True)``
            block on exit)."""
        assert example_inputs is not None or example_kwargs is not None, \
            "prepare_functional_quantization requires example_inputs and/or example_kwargs"
        builder = _FunctionalQuantBuilder(model, quant_map)
        return builder.build(example_inputs, example_kwargs)

    class functional_quantization_mode(_FunctionalQuantHooks, TorchFunctionMode):
        """Application-phase context manager for functional quantization.

        Given a :class:`FunctionalQuantState` produced by
        :func:`prepare_functional_quantization`, this context manager intercepts
        torch function calls via ``TorchFunctionMode`` and routes each intercepted
        argument to the quantizer that was created for that call site during
        preparation. It performs no quantizer creation: it only looks up and
        applies pre-created quantizers.

        Weight/parameter arguments are quantized by the persistent parametrizations
        registered during preparation, so they are not handled here.

        The ``quant_map`` spec (defined at preparation time) supports the following
        shapes. A spec is either a single spec element or a tuple of spec elements
        (one per positional argument). A spec element is one of:

        - A brevitas quantizer class, or
        - None, to skip quantization of that argument while still tracking the
          function, or
        - A resolver callable ``(module, module_name, index) -> Optional[Type]``
          returning a quantizer class (or None to skip), or
        - A ``(quantizer, di_kwargs)`` tuple, where ``di_kwargs`` are
          dependency-injection overrides applied via ``quantizer.let(**di_kwargs)``.
          Functional quantization does not infer op metadata: attributes such as
          ``group_dim`` (groupwise quantization) and ``output_channel_dim``
          (per-channel/groupwise weight quantization) must be supplied here.

        For binary functions called with exactly two positional tensor arguments, a
        3-element tuple ``(arg0_runtime_quant, arg1_runtime_quant, arg1_weight_quant)``
        selects the second argument's quantizer based on whether it is a runtime
        tensor or a parameter. Otherwise a tuple is interpreted positionally (e.g.
        query/key/value for ``scaled_dot_product_attention``).

        Args:
            state: The state returned by :func:`prepare_functional_quantization`.
            enabled: Whether quantization is active. Defaults to True.
            remove_parametrizations_on_exit: When True, the weight parametrizations
                held by ``state`` are removed on ``__exit__`` (one-shot usage). When
                False (default), they are left in place so the same ``state`` can be
                reused across multiple ``with`` blocks (e.g. calibration, then
                training, then evaluation); call
                :meth:`FunctionalQuantState.remove_parametrizations` to tear them
                down."""

        def __init__(
                self,
                state: FunctionalQuantState,
                enabled: bool = True,
                remove_parametrizations_on_exit: bool = False) -> None:
            super().__init__()
            self.state = state
            self.model = state.model
            self.enabled = enabled
            self.remove_parametrizations_on_exit = remove_parametrizations_on_exit
            self._module_stack: List[Tuple[str, nn.Module]] = []
            self._counters: Dict[str, Dict[Callable, int]] = defaultdict(lambda: defaultdict(int))
            self._hook_handles: List[RemovableHandle] = []

        def __enter__(self) -> 'functional_quantization_mode':
            self._attach_hooks()
            return super().__enter__()

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            result = super().__exit__(exc_type, exc_val, exc_tb)
            self._remove_hooks()
            if self.remove_parametrizations_on_exit:
                self.state.remove_parametrizations()
            return result

        def _lookup_act_quantizer(
                self, module_name: str, func: Callable, index: int, arg_idx: int) -> nn.Module:
            """Look up a pre-created activation quantizer, or fail fast."""
            suffix = '' if arg_idx == 0 else f'_arg{arg_idx}'
            key = _make_quantizer_key(module_name, func, index, suffix=suffix)
            quantizer = self.state.quantizers.get(key)
            if quantizer is None:
                raise RuntimeError(
                    f"No prepared quantizer found for call site '{key}'. This call was not "
                    f"seen during prepare_functional_quantization; ensure the example inputs "
                    f"exercise this path.")
            return quantizer

        def _fallback_spec_index(self, func: Callable) -> Optional[int]:
            for idx, spec_element in enumerate(self.state.arg_quant_map.get(func, [])):
                if spec_element is not None:
                    return idx
            return None

        def _effective_spec_index(
                self, func: Callable, arg_idx: int, num_args: int, is_param: bool) -> Optional[int]:
            spec_elements = self.state.arg_quant_map.get(func, [])
            if arg_idx >= len(spec_elements):
                return None
            if num_args == 2 and len(spec_elements) == 3 and arg_idx == 1:
                return 2 if is_param else 1
            return arg_idx

        def _quantize_arg(
                self,
                args: List[Any],
                arg_idx: int,
                func: Callable,
                current_module: nn.Module,
                current_module_name: str,
                index: int) -> None:
            """Apply the pre-created quantizer for a single positional argument."""
            if arg_idx >= len(args):
                return
            arg = args[arg_idx]
            if isinstance(arg, QuantTensor):
                return
            if not isinstance(arg, torch.Tensor):
                return

            is_param = (
                isinstance(arg, nn.Parameter) or arg.data_ptr() in self.state.param_to_module)
            spec_idx = self._effective_spec_index(func, arg_idx, len(args), is_param)
            spec_element = None if spec_idx is None else self.state.arg_quant_map[func][spec_idx]
            quant_class = _resolve_spec_element(
                spec_element, current_module, current_module_name, index)

            if arg_idx == 0:
                if quant_class is not None:
                    quantizer = self._lookup_act_quantizer(
                        current_module_name, func, index, arg_idx)
                    args[arg_idx] = quantizer(arg)
            else:
                param_slot = (current_module_name, func, arg_idx)
                if param_slot in self.state.parametrized_slots:
                    # Handled by the persistent weight parametrization.
                    pass
                elif is_param and quant_class is not None:
                    # A parameter that should be weight-quantized must have been
                    # parametrized during preparation; if not, fail fast.
                    raise RuntimeError(
                        f"Parameter argument at index {arg_idx} of "
                        f"{getattr(func, '__name__', func)} in module "
                        f"'{current_module_name}' was not parametrized during "
                        f"prepare_functional_quantization.")
                elif not is_param:
                    effective_class = quant_class
                    if effective_class is None:
                        fallback_idx = self._fallback_spec_index(func)
                        fallback_element = (
                            None if fallback_idx is None else
                            self.state.arg_quant_map[func][fallback_idx])
                        effective_class = _resolve_spec_element(
                            fallback_element, current_module, current_module_name, index)
                    if effective_class is not None:
                        quantizer = self._lookup_act_quantizer(
                            current_module_name, func, index, arg_idx)
                        args[arg_idx] = quantizer(arg)

        def __torch_function__(
                self,
                func: Callable,
                types: Tuple[Type, ...],
                args: Tuple[Any, ...] = (),
                kwargs: Optional[Dict[str, Any]] = None) -> Any:
            if kwargs is None:
                kwargs = {}

            if not self.enabled or func not in self.state.quant_map or not self._module_stack:
                return func(*args, **kwargs)

            current_module_name, current_module = self._module_stack[-1]
            index = self._counters[current_module_name][func]
            self._counters[current_module_name][func] += 1

            args = list(args)
            num_quant_args = len(self.state.arg_quant_map.get(func, []))
            for arg_idx in range(min(num_quant_args, len(args))):
                self._quantize_arg(args, arg_idx, func, current_module, current_module_name, index)
            return func(*tuple(args), **kwargs)

        def checkpoint_context_fn(self) -> Callable[[], Tuple[Any, Any]]:
            """Return a ``context_fn`` for ``torch.utils.checkpoint.checkpoint``.

            Gradient (activation) checkpointing recomputes the checkpointed forward
            during the backward pass inside checkpoint's own ``recompute_context``,
            which is isolated from the ``TorchFunctionMode`` entered by this context
            manager. As a result, the recompute would run *unquantized* while the
            original forward is quantized, and ``torch.utils.checkpoint`` raises
            because a different number of tensors is saved in each pass.

            Passing the returned ``context_fn`` to ``checkpoint`` re-applies this
            mode's quantization during the recompute::

                state = prepare_functional_quantization(model, quant_map, example_inputs=(x,))
                with functional_quantization_mode(state) as cm:
                    out = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False,
                        context_fn=cm.checkpoint_context_fn())
                    out.sum().backward()

            The ``context_fn`` returns ``(forward_context, recompute_context)``. The
            forward is already intercepted by this context manager (which is active
            for the whole ``with`` block), so the ``forward_context`` is a no-op to
            avoid double interception; only the ``recompute_context`` re-applies
            quantization. The recompute interceptor delegates to this instance, so
            it reuses the same module stack, call counters, and (pre-created)
            quantizers/parametrizations.

            For HuggingFace models, forward it via the checkpointing kwargs::

                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        'use_reentrant': False,
                        'context_fn': cm.checkpoint_context_fn()})
            """

            def context_fn() -> Tuple[Any, '_FunctionalQuantInterceptor']:
                return contextlib.nullcontext(), _FunctionalQuantInterceptor(self)

            return context_fn

    class _FunctionalQuantInterceptor(TorchFunctionMode):
        """Lightweight ``TorchFunctionMode`` that re-applies a parent
        :class:`functional_quantization_mode`'s quantization.

        Used as the ``recompute_context`` for ``torch.utils.checkpoint`` so that
        quantization is applied during the checkpoint recompute (the original
        forward is already intercepted by the parent context manager). Unlike
        :class:`functional_quantization_mode`, entering/exiting this mode only
        pushes/pops the torch-function mode; it does not attach hooks or remove
        parametrizations, because that state is owned by the parent and remains
        alive for the whole training step."""

        def __init__(self, parent: 'functional_quantization_mode') -> None:
            super().__init__()
            self._parent = parent

        def __torch_function__(
                self,
                func: Callable,
                types: Tuple[Type, ...],
                args: Tuple[Any, ...] = (),
                kwargs: Optional[Dict[str, Any]] = None) -> Any:
            return functional_quantization_mode.__torch_function__(
                self._parent, func, types, args, kwargs)

else:
    functional_quantization_mode = object()
    FunctionalQuantState = object()

    def prepare_functional_quantization(*args, **kwargs):
        raise RuntimeError("Functional quantization requires PyTorch >= 1.12")


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
