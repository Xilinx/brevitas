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
    # A single spec element is either a quantizer class, a resolver callable, or None.
    QuantSpecElement = Optional[Union[Type, QuantResolver]]
    QuantSpecType = Union[QuantSpecElement, Tuple[QuantSpecElement, ...]]

    class _FunctionalQuantModuleBase(nn.Module):
        """Base shim used to preserve op metadata for functional quantization.

        Dependency injection in Brevitas resolves several quantizer properties from the
        tracked module. Functional quantization creates quantizers outside of concrete
        quant layer modules, so this shim mirrors the relevant metadata from the
        intercepted functional op.
        """

        def __init__(self, op_metadata: Dict[str, Any]) -> None:
            super().__init__()
            self.bias = None
            self._op_metadata = dict(op_metadata)
            self.functional_layer_kind = self._op_metadata.get('layer_kind')
            self.transposed = self._op_metadata.get('transposed', False)

        @property
        def output_channel_dim(self) -> int:
            return self._op_metadata.get('output_channel_dim', 0)

        @property
        def channelwise_separable(self) -> bool:
            return False

        @property
        def per_elem_ops(self) -> int:
            return 0

    class _FunctionalActQuantModule(_FunctionalQuantModuleBase):
        """Shim module for runtime activation quantization metadata only."""

        def __init__(self, op_metadata: Dict[str, Any]) -> None:
            super().__init__(op_metadata)

    class _FunctionalWeightQuantModule(_FunctionalQuantModuleBase):
        """Shim module for weight quantization metadata and weight-shaped properties."""

        def __init__(self, weight: Tensor, op_metadata: Dict[str, Any]) -> None:
            super().__init__(op_metadata)
            self.weight = weight

        @property
        def out_channels(self) -> int:
            return self.weight.shape[self.output_channel_dim]

    class _FunctionalActQuantIdentity(QuantIdentity):
        """QuantIdentity wrapper that exposes functional-op metadata to DI solvers."""

        def __init__(
                self, act_quant: Type, tracked_module: _FunctionalActQuantModule, **kwargs) -> None:
            self.functional_layer_kind = tracked_module.functional_layer_kind
            self.transposed = tracked_module.transposed
            self._functional_output_channel_dim = tracked_module.output_channel_dim
            super().__init__(act_quant=act_quant, **kwargs)

        @property
        def output_channel_dim(self) -> int:
            return self._functional_output_channel_dim

    class _WeightQuantHolder(nn.Module):
        """Dummy module that holds a weight parameter so that
        WeightQuantProxyFromInjector can be instantiated through the standard
        QuantProxyMixin path"""

        def __init__(
                self, weight: nn.Parameter, tracked_module: _FunctionalWeightQuantModule) -> None:
            super().__init__()
            self.weight = weight
            self.tracked_module = tracked_module

        @property
        def functional_layer_kind(self) -> Optional[str]:
            return self.tracked_module.functional_layer_kind

        @property
        def transposed(self) -> bool:
            return self.tracked_module.transposed

        @property
        def output_channel_dim(self) -> int:
            return self.tracked_module.output_channel_dim

        @property
        def out_channels(self) -> int:
            return self.tracked_module.out_channels

    class _QuantParametrization(nn.Module):
        """Parametrization module that quantizes a parameter (e.g., weight) on-the-fly
        during forward using a weight quantization proxy."""

        def __init__(self, weight_quant_proxy: nn.Module) -> None:
            super().__init__()
            self.weight_quant_proxy = weight_quant_proxy

        def forward(self, x: Tensor) -> Tensor:
            out = self.weight_quant_proxy(x)
            return _unpack_quant_tensor(out)

    class functional_quantization_mode(TorchFunctionMode):
        """Context manager that uses hooks and TorchFunctionMode to quantize inputs to
        torch functions without requiring specialized PassThrough modules.

        Args:
            model: The nn.Module whose forward pass will be intercepted.
            quant_map: A mapping from torch functions (e.g. torch.nn.functional.linear)
                to a spec. A spec is either a single spec element or a tuple of spec
                elements (one per positional argument). A spec element is one of:
                - A brevitas quantizer class, or
                - None, to skip quantization of that argument while still tracking
                  the function, or
                - A resolver callable with signature
                  ``(module, module_name, index) -> Optional[Type]`` that returns a
                  quantizer class (or None to skip). ``module`` is the current
                  nn.Module instance, ``module_name`` is its name, and ``index`` is
                  the per-(module, func) call index. Resolvers may appear in any
                  position, mixed with quantizer classes and None. The quantizer
                  returned by a resolver is created once on first encounter and
                  reused on subsequent forwards (resolvers should be deterministic).
                The supported spec shapes are:
                - A single spec element for the first input tensor.
                - A tuple of spec elements, one per positional argument.
                - For binary functions (called with exactly two positional tensor
                  arguments), a 3-element tuple with the form
                  ``(arg0_runtime_quant, arg1_runtime_quant, arg1_weight_quant)``.
                  In that case, the second argument uses the runtime quantizer when
                  it is a regular tensor and the weight quantizer when it is a
                  parameter. This dispatch only applies when the call has exactly two
                  positional arguments; otherwise the tuple is interpreted
                  positionally (e.g. query/key/value for
                  ``scaled_dot_product_attention``).
                For arguments at index >= 1:
                    - If the argument is a parameter, a weight-style quantizer is
                      registered as a persistent parametrization.
                    - If the argument is a regular tensor (not a parameter and not
                      already quantized), a QuantIdentity activation quantizer is
                      created using the explicit spec element (or falling back to the
                      first non-None spec element in the tuple).
            enabled: Whether quantization is active. Defaults to True.
        """

        def __init__(
                self,
                model: nn.Module,
                quant_map: Dict[Callable, QuantSpecType],
                enabled: bool = True) -> None:
            super().__init__()
            self.model = model
            self.enabled = enabled
            # Stack of (module_name, module) to track which nn.Module we are in
            self._module_stack: List[Tuple[str, nn.Module]] = []
            # Per-module, per-function call counters: {module_name: {func: int}}
            self._counters: Dict[str, Dict[Callable, int]] = defaultdict(lambda: defaultdict(int))
            # Quantizer registry: {key: nn.Module}
            self._quantizers: Dict[str, nn.Module] = {}
            # Hook handles for cleanup
            self._hook_handles: List[RemovableHandle] = []
            # Whether quantizers have been initialized (first pass done)
            self._initialized = False
            # Mapping from parameter data_ptr to (module, param_name) for parametrization
            self._param_to_module: Dict[int, Tuple[nn.Module, str]] = {}
            # Track parametrizations we registered so we can remove them on exit
            self._registered_parametrizations: List[Tuple[nn.Module, str]] = []
            # (module_name, func, arg_idx) slots that already have a weight
            # parametrization, so recompute passes do not re-quantize them.
            self._parametrized_slots: Set[Tuple[str, Callable, int]] = set()

            # Set of all functions we should intercept
            self._quant_map: Dict[Callable, QuantSpecType] = {}
            # Per-function list of spec elements per positional arg. Each element is a
            # quantizer class, a resolver callable, or None.
            self._arg_quant_map: Dict[Callable, List[QuantSpecElement]] = {}
            for func, spec in quant_map.items():
                self._quant_map[func] = spec
                if isinstance(spec, tuple):
                    self._arg_quant_map[func] = list(spec)
                else:
                    self._arg_quant_map[func] = [spec]

        def _make_quantizer_key(
                self, module_name: str, func: Callable, index: int, suffix: str = '') -> str:
            """Create a unique key for a quantizer instance.

            Dots in module_name are replaced with underscores because
            ``nn.Module.add_module`` does not allow dots in names."""
            func_name = getattr(func, '__name__', str(func))
            safe_name = module_name.replace('.', '_')
            return f'_fq_{safe_name}_{func_name}_{index}{suffix}'

        def _move_to_model_device(self, module: nn.Module) -> nn.Module:
            """Move a module to the same device as the model."""
            try:
                device = next(self.model.parameters()).device
                module = module.to(device)
            except StopIteration:
                pass
            return module

        def _functional_op_metadata(self, func: Callable, arg_idx: int = 0) -> Dict[str, Any]:
            """Build functional-op metadata consumed by dependency-injection solvers."""
            transposed_funcs = {
                torch.nn.functional.conv_transpose1d,
                torch.nn.functional.conv_transpose2d,
                torch.nn.functional.conv_transpose3d}
            if func == torch.nn.functional.linear:
                return {'layer_kind': 'linear', 'output_channel_dim': 0, 'transposed': False}
            elif func in {torch.nn.functional.conv1d,
                          torch.nn.functional.conv2d,
                          torch.nn.functional.conv3d}:
                return {'layer_kind': 'conv', 'output_channel_dim': 0, 'transposed': False}
            elif func in transposed_funcs:
                return {'layer_kind': 'conv', 'output_channel_dim': 1, 'transposed': True}
            elif func in {torch.bmm, torch.matmul}:
                return {'layer_kind': 'linear', 'output_channel_dim': 0, 'transposed': False}
            elif func == torch.nn.functional.scaled_dot_product_attention:
                if arg_idx == 1:
                    return {'layer_kind': 'linear', 'output_channel_dim': 0, 'transposed': True}
                return {'layer_kind': 'linear', 'output_channel_dim': 0, 'transposed': False}
            return {'layer_kind': None, 'output_channel_dim': 0, 'transposed': False}

        def _make_act_tracked_module(
                self, func: Callable, arg_idx: int) -> _FunctionalActQuantModule:
            op_metadata = self._functional_op_metadata(func, arg_idx)
            return _FunctionalActQuantModule(op_metadata)

        def _make_weight_tracked_module(
                self, func: Callable, tensor: Tensor, arg_idx: int) -> _FunctionalWeightQuantModule:
            op_metadata = self._functional_op_metadata(func, arg_idx)
            return _FunctionalWeightQuantModule(tensor, op_metadata)

        def _create_act_quantizer(
                self, quant_class: Type, func: Callable, tensor: Tensor,
                arg_idx: int) -> QuantIdentity:
            """Create a QuantIdentity quantizer for activation tensors."""
            del tensor
            tracked_module = self._make_act_tracked_module(func, arg_idx)
            quantizer = _FunctionalActQuantIdentity(
                act_quant=quant_class, tracked_module=tracked_module, return_quant_tensor=True)
            quantizer.train(self.model.training)
            return self._move_to_model_device(quantizer)

        def _create_weight_quant_proxy(
                self, quant_class: Type, weight_param: nn.Parameter, func: Callable,
                arg_idx: int) -> nn.Module:
            """Create a weight quantization proxy through an op-aware holder module."""
            tracked_module = self._make_weight_tracked_module(func, weight_param, arg_idx)
            holder = _WeightQuantHolder(weight_param, tracked_module)
            quant_injector = quant_class.let()
            proxy = quant_injector.proxy_class(holder, quant_injector)
            return self._move_to_model_device(proxy)

        def _get_or_create_act_quantizer(
                self,
                module_name: str,
                func: Callable,
                index: int,
                arg_idx: int,
                quant_class: Type,
                arg: Tensor) -> QuantIdentity:
            """Get an existing activation quantizer or create a new one.

            Args:
                module_name: Name of the current nn.Module.
                func: The torch function being intercepted.
                index: Call index within this module for this function.
                arg_idx: Positional argument index (0 = first arg, 1 = second, etc.).
                quant_class: The quantizer class to use.
                arg: Runtime tensor argument used to build the tracked-module metadata.
            """
            suffix = '' if arg_idx == 0 else f'_arg{arg_idx}'
            key = self._make_quantizer_key(module_name, func, index, suffix=suffix)
            if key not in self._quantizers:
                quantizer = self._create_act_quantizer(quant_class, func, arg, arg_idx)
                self._quantizers[key] = quantizer
                if not hasattr(self.model, key):
                    self.model.add_module(key, quantizer)
            return self._quantizers[key]

        def _build_param_to_module_map(self) -> None:
            """Build a mapping from parameter data_ptr to (module, param_name)."""
            self._param_to_module.clear()
            for name, module in self.model.named_modules():
                if name.startswith('_fq_'):
                    continue
                for param_name, param in module.named_parameters(recurse=False):
                    self._param_to_module[param.data_ptr()] = (module, param_name)

        def _register_weight_parametrization(
                self,
                param_tensor: nn.Parameter,
                func: Callable,
                module_name: str,
                index: int,
                arg_idx: int,
                quant_class: Type) -> None:
            """Register a quantization parametrization on the module that owns the parameter.

            Creates an op-aware weight quant proxy, wraps it in a ``_QuantParametrization``,
            and registers it on the parameter-owning module for the lifetime of the
            context manager.

            Args:
                param_tensor: The parameter tensor to parametrize.
                func: The torch function being intercepted.
                module_name: Name of the current nn.Module.
                index: Call index within this module for this function.
                arg_idx: Positional argument index.
                quant_class: The weight quantizer class to use.
            """
            data_ptr = param_tensor.data_ptr()
            if data_ptr not in self._param_to_module:
                return
            owner_module, param_name = self._param_to_module[data_ptr]
            # Check if this parametrization was already registered by us
            if is_parametrized(owner_module, param_name):
                for p in getattr(owner_module.parametrizations, param_name):
                    if isinstance(p, _QuantParametrization):
                        return  # Already registered

            # Create weight quant proxy through the standard path
            weight_quant_proxy = self._create_weight_quant_proxy(
                quant_class, param_tensor, func, arg_idx)

            # Store for state tracking
            suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
            key = self._make_quantizer_key(module_name, func, index, suffix=suffix)
            self._quantizers[key] = weight_quant_proxy
            if not hasattr(self.model, key):
                self.model.add_module(key, weight_quant_proxy)

            param_module = _QuantParametrization(weight_quant_proxy)
            register_parametrization(owner_module, param_name, param_module)
            self._registered_parametrizations.append((owner_module, param_name))

        def _remove_parametrizations(self) -> None:
            """Remove all parametrizations we registered."""
            for owner_module, param_name in self._registered_parametrizations:
                if is_parametrized(owner_module, param_name):
                    remove_parametrizations(owner_module, param_name, leave_parametrized=True)
            self._registered_parametrizations.clear()
            self._parametrized_slots.clear()

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
                # Skip quantizer modules we added ourselves
                if name.startswith('_fq_'):
                    continue
                pre_handle = module.register_forward_pre_hook(self._pre_hook(name))
                post_handle = module.register_forward_hook(self._post_hook(name))
                self._hook_handles.append(pre_handle)
                self._hook_handles.append(post_handle)
            # Attach a hook on the top-level model to reset counters after each forward
            def reset_hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> None:
                self._reset_counters()
                self._initialized = True

            handle = self.model.register_forward_hook(reset_hook)
            self._hook_handles.append(handle)

        def _remove_hooks(self) -> None:
            """Remove all attached hooks."""
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

        def __enter__(self) -> 'functional_quantization_mode':
            self._build_param_to_module_map()
            self._attach_hooks()
            return super().__enter__()

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            result = super().__exit__(exc_type, exc_val, exc_tb)
            self._remove_hooks()
            self._remove_parametrizations()
            return result

        def _is_quant_tensor(self, t: Any) -> bool:
            """Check if a value is already a QuantTensor."""
            return isinstance(t, QuantTensor)

        def _resolve_spec_element(
                self, elem: 'QuantSpecElement', module: nn.Module, module_name: str,
                index: int) -> Optional[Type]:
            """Resolve a spec element to a concrete quantizer class.

            A spec element is either ``None``, a quantizer class, or a resolver
            callable. Resolver callables are invoked as
            ``elem(module, module_name, index)`` and must return a quantizer class
            (or ``None`` to skip quantization of the argument). Quantizer classes
            are returned unchanged.
            """
            if elem is None:
                return None
            # Quantizer classes are types; resolver lambdas are not.
            if isinstance(elem, type):
                return elem
            if callable(elem):
                return elem(module, module_name, index)
            return elem

        def _fallback_spec_element(self, func: Callable) -> 'QuantSpecElement':
            """Return the first non-None spec element in the arg list for ``func``."""
            for spec_element in self._arg_quant_map.get(func, []):
                if spec_element is not None:
                    return spec_element
            return None

        def _effective_spec_element(
                self, func: Callable, arg_idx: int, num_args: int,
                is_param: bool) -> 'QuantSpecElement':
            """Select the spec element to use for a positional argument.

            By default, quantizer specs are positional. For binary functions only,
            a 3-element spec is also supported with the layout
            ``(arg0_runtime_quant, arg1_runtime_quant, arg1_weight_quant)`` so that
            the second argument can use a different quantizer depending on whether it
            is a runtime tensor or a parameter.

            The returned value is a raw spec element (quantizer class, resolver
            callable, or ``None``); use ``_resolve_spec_element`` to obtain a
            concrete quantizer class.
            """
            spec_elements = self._arg_quant_map.get(func, [])
            if arg_idx >= len(spec_elements):
                return None
            if num_args == 2 and len(spec_elements) == 3 and arg_idx == 1:
                return spec_elements[2] if is_param else spec_elements[1]
            return spec_elements[arg_idx]

        def _quantize_arg(
                self,
                args: List[Any],
                arg_idx: int,
                func: Callable,
                current_module: nn.Module,
                current_module_name: str,
                index: int) -> None:
            """Quantize a single positional argument in-place within *args*.

            For ``arg_idx == 0`` an activation quantizer is used when specified.
            For ``arg_idx >= 1`` the argument is handled according to its runtime type:
            existing ``QuantTensor`` instances are skipped, parameters use persistent weight
            parametrization, and regular tensors use activation quantization.

            Spec elements may be resolver callables; they are resolved to a concrete
            quantizer class via ``_resolve_spec_element`` using the current module
            instance, its name, and the call index.
            """
            if arg_idx >= len(args):
                return
            arg = args[arg_idx]
            if self._is_quant_tensor(arg):
                return
            if not isinstance(arg, torch.Tensor):
                return

            is_param = isinstance(arg, nn.Parameter) or arg.data_ptr() in self._param_to_module
            spec_element = self._effective_spec_element(func, arg_idx, len(args), is_param)
            quant_class = self._resolve_spec_element(
                spec_element, current_module, current_module_name, index)

            if arg_idx == 0:
                # First argument: always use an activation quantizer
                if quant_class is not None:
                    quantizer = self._get_or_create_act_quantizer(
                        current_module_name, func, index, arg_idx, quant_class, arg)
                    args[arg_idx] = quantizer(arg)
            else:
                # Subsequent arguments: handle parameter vs non-parameter
                wq_suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
                wq_key = self._make_quantizer_key(
                    current_module_name, func, index, suffix=wq_suffix)
                # A weight parametrization for this (module, func, arg position) is
                # registered at most once. Once registered, the owning module's
                # parameter is quantized by the parametrization itself, so this
                # argument must not be quantized again. This check is
                # index-independent so that a gradient-checkpointing recompute
                # (which may see a different call index, or a non-parameter tensor
                # that is actually the parametrized weight) does not re-quantize it.
                param_slot = (current_module_name, func, arg_idx)
                already_parametrized = param_slot in self._parametrized_slots
                if already_parametrized:
                    # The parametrization has already quantized this tensor
                    pass
                elif is_param and quant_class is not None:
                    # Arg is a parameter and a weight quantizer was specified:
                    # register a persistent parametrization on the owning module.
                    # On this first call the parametrization wasn't active yet, so
                    # we must also quantize explicitly. On subsequent forwards the
                    # parametrization handles it automatically.
                    self._register_weight_parametrization(
                        arg, func, current_module_name, index, arg_idx, quant_class)
                    self._parametrized_slots.add(param_slot)
                    weight_quant_proxy = self._quantizers[wq_key]
                    args[arg_idx] = weight_quant_proxy(arg)
                elif not is_param:
                    # Arg is a regular tensor (not a parameter).
                    # Use the explicit quantizer class if provided,
                    # otherwise fall back to the first non-None spec element.
                    effective_class = quant_class
                    if effective_class is None:
                        effective_class = self._resolve_spec_element(
                            self._fallback_spec_element(func),
                            current_module,
                            current_module_name,
                            index)
                    if effective_class is not None:
                        quantizer = self._get_or_create_act_quantizer(
                            current_module_name, func, index, arg_idx, effective_class, arg)
                        args[arg_idx] = quantizer(arg)

        def __torch_function__(
                self,
                func: Callable,
                types: Tuple[Type, ...],
                args: Tuple[Any, ...] = (),
                kwargs: Optional[Dict[str, Any]] = None) -> Any:
            if kwargs is None:
                kwargs = {}

            if not self.enabled or func not in self._quant_map or not self._module_stack:
                return func(*args, **kwargs)

            # Identify the current (innermost) module
            current_module_name, current_module = self._module_stack[-1]

            # Get and increment the call counter for this (module, func) pair
            index = self._counters[current_module_name][func]
            self._counters[current_module_name][func] += 1

            args = list(args)

            # Quantize each positional argument that has a quantizer spec
            num_quant_args = len(self._arg_quant_map.get(func, []))
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

                with functional_quantization_mode(model, quant_map) as cm:
                    # Calibrate first: no recompute happens under no_grad, so the
                    # quantizers are created once here.
                    with torch.no_grad():
                        model(calibration_batch)
                    out = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False,
                        context_fn=cm.checkpoint_context_fn())
                    out.sum().backward()

            The ``context_fn`` returns ``(forward_context, recompute_context)``. The
            forward is already intercepted by this context manager (which is active
            for the whole ``with`` block), so the ``forward_context`` is a no-op to
            avoid double interception; only the ``recompute_context`` re-applies
            quantization. The recompute interceptor delegates to this instance, so
            it reuses the same module stack, call counters, quantizers, and
            parametrizations created during the forward. It intentionally does not
            manage hooks or parametrizations: that lifecycle is owned by this
            context manager for the whole duration of the ``with`` block.

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
        :class:`functional_quantization_mode`,
        entering/exiting this mode only pushes/pops the torch-function mode; it
        does not attach hooks, build the parameter map, or register/remove
        parametrizations, because that state is owned by the parent and remains
        alive for the whole training step.
        """

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
