# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

from packaging import version
import torch
from torch import nn

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

    class _WeightQuantHolder(nn.Module):
        """Dummy module that holds a weight parameter so that
        WeightQuantProxyFromInjector can be instantiated through the standard
        QuantProxyMixin path"""

        def __init__(self, weight: nn.Parameter):
            super().__init__()
            self.weight = weight

    class _QuantParametrization(nn.Module):
        """Parametrization module that quantizes a parameter (e.g., weight) on-the-fly
        during forward using a weight quantization proxy."""

        def __init__(self, weight_quant_proxy):
            super().__init__()
            self.weight_quant_proxy = weight_quant_proxy

        def forward(self, x):
            out = self.weight_quant_proxy(x)
            return _unpack_quant_tensor(out)

    class functional_quantization_mode(TorchFunctionMode):
        """Context manager that uses hooks and TorchFunctionMode to quantize inputs to
        torch functions without requiring specialized PassThrough modules.

        Args:
            model: The nn.Module whose forward pass will be intercepted.
            quant_map: A mapping from torch functions (e.g. torch.nn.functional.linear)
                to either:
                - A brevitas quantizer class for the first input tensor (or None to
                  skip first-input quantization while still tracking the function), or
                - A tuple of quantizer classes, one per positional argument.
                  Each element may be None to skip quantization of that argument.
                  For arguments at index >= 1:
                    - If the argument is a parameter, a weight-style quantizer is
                      registered as a persistent parametrization.
                    - If the argument is a regular tensor (not a parameter and not
                      already quantized), a QuantIdentity activation quantizer is
                      created using the explicit class (or falling back to the first
                      non-None quantizer class in the tuple).
            enabled: Whether quantization is active. Defaults to True.
        """

        def __init__(self, model: torch.nn.Module, quant_map: Dict, enabled: bool = True):
            super().__init__()
            self.model = model
            self.enabled = enabled
            # Stack of (module_name, module) to track which nn.Module we are in
            self._module_stack = []
            # Per-module, per-function call counters: {module_name: {func: int}}
            self._counters: Dict = defaultdict(lambda: defaultdict(int))
            # Quantizer registry: {key: nn.Module}
            self._quantizers: Dict = {}
            # Hook handles for cleanup
            self._hook_handles = []
            # Whether quantizers have been initialized (first pass done)
            self._initialized = False
            # Mapping from parameter data_ptr to (module, param_name) for parametrization
            self._param_to_module: Dict = {}
            # Track parametrizations we registered so we can remove them on exit
            self._registered_parametrizations = []

            # Set of all functions we should intercept
            self._quant_map: Dict = {}
            # Per-function list of quantizer classes per positional arg (may contain None)
            self._arg_quant_map: Dict = {}
            for func, spec in quant_map.items():
                self._quant_map[func] = spec
                if isinstance(spec, tuple):
                    self._arg_quant_map[func] = list(spec)
                else:
                    self._arg_quant_map[func] = [spec]

        def _make_quantizer_key(self, module_name, func, index, suffix=''):
            """Create a unique key for a quantizer instance.

            Dots in module_name are replaced with underscores because
            ``nn.Module.add_module`` does not allow dots in names."""
            func_name = getattr(func, '__name__', str(func))
            safe_name = module_name.replace('.', '_')
            return f'_fq_{safe_name}_{func_name}_{index}{suffix}'

        def _move_to_model_device(self, module):
            """Move a module to the same device as the model."""
            try:
                device = next(self.model.parameters()).device
                module = module.to(device)
            except StopIteration:
                pass
            return module

        def _create_act_quantizer(self, quant_class):
            """Create a QuantIdentity quantizer for activation tensors."""
            quantizer = QuantIdentity(act_quant=quant_class, return_quant_tensor=True)
            quantizer.train(self.model.training)
            return self._move_to_model_device(quantizer)

        def _create_weight_quant_proxy(self, quant_class, weight_param):
            """Create a weight quantization proxy through a dummy _WeightQuantHolder module.

            This follows the standard QuantProxyMixin instantiation path (Option B):
            the quant_class resolver needs a tracked module with a .weight attribute."""
            holder = _WeightQuantHolder(weight_param)
            quant_injector = quant_class.let()
            proxy = quant_injector.proxy_class(holder, quant_injector)
            return self._move_to_model_device(proxy)

        def _get_or_create_act_quantizer(self, module_name, func, index, arg_idx, quant_class):
            """Get an existing activation quantizer or create a new one.

            Args:
                module_name: Name of the current nn.Module.
                func: The torch function being intercepted.
                index: Call index within this module for this function.
                arg_idx: Positional argument index (0 = first arg, 1 = second, etc.).
                quant_class: The quantizer class to use.
            """
            suffix = '' if arg_idx == 0 else f'_arg{arg_idx}'
            key = self._make_quantizer_key(module_name, func, index, suffix=suffix)
            if key not in self._quantizers:
                quantizer = self._create_act_quantizer(quant_class)
                self._quantizers[key] = quantizer
                if not hasattr(self.model, key):
                    self.model.add_module(key, quantizer)
            return self._quantizers[key]

        def _build_param_to_module_map(self):
            """Build a mapping from parameter data_ptr to (module, param_name)."""
            self._param_to_module.clear()
            for name, module in self.model.named_modules():
                if name.startswith('_fq_'):
                    continue
                for param_name, param in module.named_parameters(recurse=False):
                    self._param_to_module[param.data_ptr()] = (module, param_name)

        def _register_weight_parametrization(
                self, param_tensor, func, module_name, index, arg_idx, quant_class):
            """Register a quantization parametrization on the module that owns the parameter.

            Creates a weight quant proxy through _WeightQuantHolder and wraps it in
            a _QuantParametrization. The parametrization persists for the lifetime of
            the context manager.

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
            weight_quant_proxy = self._create_weight_quant_proxy(quant_class, param_tensor)

            # Store for state tracking
            suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
            key = self._make_quantizer_key(module_name, func, index, suffix=suffix)
            self._quantizers[key] = weight_quant_proxy
            if not hasattr(self.model, key):
                self.model.add_module(key, weight_quant_proxy)

            param_module = _QuantParametrization(weight_quant_proxy)
            register_parametrization(owner_module, param_name, param_module)
            self._registered_parametrizations.append((owner_module, param_name))

        def _remove_parametrizations(self):
            """Remove all parametrizations we registered."""
            for owner_module, param_name in self._registered_parametrizations:
                if is_parametrized(owner_module, param_name):
                    remove_parametrizations(owner_module, param_name, leave_parametrized=True)
            self._registered_parametrizations.clear()

        def _pre_hook(self, module_name):
            """Create a forward pre-hook that pushes the module onto the stack."""

            def hook(module, args):
                self._module_stack.append((module_name, module))

            return hook

        def _post_hook(self, module_name):
            """Create a forward hook that pops the module from the stack."""

            def hook(module, args, output):
                if self._module_stack and self._module_stack[-1][0] == module_name:
                    self._module_stack.pop()

            return hook

        def _reset_counters(self):
            """Reset all per-module call counters."""
            self._counters.clear()

        def _attach_hooks(self):
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
            def reset_hook(module, args, output):
                self._reset_counters()
                self._initialized = True

            handle = self.model.register_forward_hook(reset_hook)
            self._hook_handles.append(handle)

        def _remove_hooks(self):
            """Remove all attached hooks."""
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

        def __enter__(self):
            self._build_param_to_module_map()
            self._attach_hooks()
            return super().__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            result = super().__exit__(exc_type, exc_val, exc_tb)
            self._remove_hooks()
            self._remove_parametrizations()
            return result

        def _is_quant_tensor(self, t):
            """Check if a value is already a QuantTensor."""
            return isinstance(t, QuantTensor)

        def _fallback_quant_class(self, func):
            """Return the first non-None quantizer class in the arg list for ``func``."""
            for qc in self._arg_quant_map.get(func, []):
                if qc is not None:
                    return qc
            return None

        def _quantize_arg(self, args, arg_idx, func, current_module_name, index):
            """Quantize a single positional argument in-place within *args*.

            For arg_idx == 0 (the first argument), an activation quantizer is always used.
            For arg_idx >= 1:
              - If the arg is already a QuantTensor, skip.
              - If the arg is a parameter and a quantizer class is provided, register
                a persistent weight parametrization.
              - If the arg is a non-parameter tensor, create an activation quantizer
                using the explicit class or falling back to the first non-None class.
            """
            if arg_idx >= len(args):
                return
            arg = args[arg_idx]
            if self._is_quant_tensor(arg):
                return
            if not isinstance(arg, torch.Tensor):
                return

            quant_classes = self._arg_quant_map.get(func, [])
            quant_class = quant_classes[arg_idx] if arg_idx < len(quant_classes) else None

            if arg_idx == 0:
                # First argument: always use an activation quantizer
                if quant_class is not None:
                    quantizer = self._get_or_create_act_quantizer(
                        current_module_name, func, index, arg_idx, quant_class)
                    args[arg_idx] = quantizer(arg)
            else:
                # Subsequent arguments: handle parameter vs non-parameter
                wq_suffix = f'_arg{arg_idx}_wq' if arg_idx > 1 else '_wq'
                wq_key = self._make_quantizer_key(
                    current_module_name, func, index, suffix=wq_suffix)
                already_parametrized = wq_key in self._quantizers
                is_param = isinstance(arg, nn.Parameter) or \
                    arg.data_ptr() in self._param_to_module
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
                    weight_quant_proxy = self._quantizers[wq_key]
                    args[arg_idx] = weight_quant_proxy(arg)
                elif not is_param:
                    # Arg is a regular tensor (not a parameter).
                    # Use the explicit quantizer class if provided,
                    # otherwise fall back to the first non-None class.
                    effective_class = quant_class
                    if effective_class is None:
                        effective_class = self._fallback_quant_class(func)
                    if effective_class is not None:
                        quantizer = self._get_or_create_act_quantizer(
                            current_module_name, func, index, arg_idx, effective_class)
                        args[arg_idx] = quantizer(arg)

        def __torch_function__(self, func, types, args=(), kwargs=None):
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
                self._quantize_arg(args, arg_idx, func, current_module_name, index)

            return func(*tuple(args), **kwargs)

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
