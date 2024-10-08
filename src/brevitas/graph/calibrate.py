# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from copy import deepcopy
from functools import partial
import sys

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.nn import QuantHardTanh
from brevitas.nn import QuantLinear
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor

from .base import Transform

__all__ = [
    'ClipFloatWeights',
    'DisableEnableQuantization',
    'bias_correction_mode',
    'calibration_mode',
    'load_quant_model_mode']

_PARAM_PROXIES = (WeightQuantProxyFromInjectorBase, BiasQuantProxyFromInjectorBase)

_BIAS_PROXIES = (BiasQuantProxyFromInjectorBase)

_ACC_PROXIES = (TruncQuantProxyFromInjector, ClampQuantProxyFromInjector)

_LAYERS_TO_CLIP = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d)

BN_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def disable_return_quant_tensor(model):
    previous_state = {}
    for module in model.modules():
        if hasattr(module, 'return_quant_tensor'):
            previous_state[module] = module.return_quant_tensor
            module.return_quant_tensor = False
    return previous_state


def restore_return_quant_tensor(model, previous_state):
    for module in model.modules():
        if hasattr(module, 'return_quant_tensor') and module in previous_state:
            module.return_quant_tensor = previous_state[module]


def extend_collect_stats_steps(module):
    if hasattr(module, 'collect_stats_steps'):
        # We extend the collect steps in PTQ to match potentially long calibrations
        module.collect_stats_steps = sys.maxsize


def set_collect_stats_to_average(module):
    if hasattr(module, 'collect_stats_steps') and hasattr(module, 'momentum'):
        # Set the average for stats collection to a true average (default is EMA)
        module.momentum = None


def finalize_collect_stats(module):
    if hasattr(module, 'collect_stats_steps') and hasattr(module, 'counter'):
        # If the counter has already reached collect_stats_steps, we do not want to reset it
        # otherwise the restrict_preprocess might be applied twice: during calibration
        # (that happens in training mode) and then when the model is evaluated
        module.counter = max(module.collect_stats_steps, module.counter)


class calibration_mode:

    def __init__(self, model, enabled=True):
        self.model = model
        self.previous_training_state = model.training
        self.disable_quant_inference = DisableEnableQuantization(call_act_quantizer_impl=True)
        self.enabled = enabled
        self.return_quant_tensor_state = dict()

    def __enter__(self):
        if self.enabled:
            self.model.apply(extend_collect_stats_steps)
            self.model.apply(set_collect_stats_to_average)
            self.return_quant_tensor_state = disable_return_quant_tensor(self.model)
            self.disable_quant_inference.apply(
                self.model, is_training=True, quantization_enabled=False)

    def __exit__(self, type, value, traceback):
        self.model.apply(finalize_collect_stats)
        self.disable_quant_inference.apply(
            self.model, is_training=self.previous_training_state, quantization_enabled=True)
        restore_return_quant_tensor(self.model, self.return_quant_tensor_state)


class bias_correction_mode:

    def __init__(self, model, enabled=True, skip_if_no_bias=False):
        self.model = model
        self.bias_correction = _BiasCorrection(skip_if_no_bias=skip_if_no_bias)
        self.enabled = enabled
        self.hooks = []
        self.output_quant_modules = []

    def __enter__(self):
        if self.enabled:
            for module in self.model.modules():
                # Disable output quant so that the bias correction can be merged in the bias
                if hasattr(module, 'output_quant') and module.output_quant.is_quant_enabled:
                    self.bias_correction.disable_act_quantization(
                        module.output_quant, is_training=False)
                    self.output_quant_modules.append(module)
            self.bias_correction.register_hook_to_wbiol(self.model, self.hooks)

    def __exit__(self, type, value, traceback):
        self.bias_correction.apply_correction(self.model)
        for module in self.output_quant_modules:
            # Re-enable output quantization
            self.bias_correction.enable_act_quantization(module.output_quant, is_training=False)
        for hook in self.hooks:
            hook.remove()


class load_quant_model_mode:

    def __init__(self, model):
        self.model = model
        self.tracked_modules = []

    def __enter__(self):
        for module in self.model.modules():
            if issubclass(type(module), QuantWBIOL):
                module._quant_load_model_mode = True

    def __exit__(self, *args, **kwargs):
        for module in self.model.modules():
            if issubclass(type(module), QuantWBIOL):
                module._quant_load_model_mode = False


class ClipFloatWeights(Transform):

    def __init__(self, threshold=15., layers_to_clip=_LAYERS_TO_CLIP) -> None:
        super(ClipFloatWeights, self).__init__()
        self.threshold = threshold
        self.layers_to_clip = layers_to_clip

    def apply(self, model):
        for module in model.modules():
            if isinstance(module, self.layers_to_clip):
                module.weight.data.clamp_(-self.threshold, self.threshold)
        return model


class DisableEnableQuantization(Transform):

    def __init__(self, call_act_quantizer_impl=False):
        super(DisableEnableQuantization, self).__init__()
        self.disable_act_quant_hooks = []
        self.call_act_quantizer_impl = call_act_quantizer_impl

    def unpack_input(self, inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        if isinstance(inp, QuantTensor):
            inp = inp.value
        return inp

    def disable_act_quant_hook(self, module, inp, output):
        inp = self.unpack_input(inp)
        if module.fused_activation_quant_proxy is not None:
            inp = module.fused_activation_quant_proxy.activation_impl(inp)
        # consider the first module as representative for the activation fn
        # as this is what would happen with a shared act_quant
        # this gets called both during (empty) input_quant and act_quant
        # but for HardTanh it's not an issue
        if isinstance(module.tracked_module_list[0], QuantHardTanh):
            inp = F.hardtanh(
                inp, min_val=module.quant_injector.min_val, max_val=module.quant_injector.max_val)
        return inp

    def disable_act_quantization(self, model, is_training):
        # If self.call_act_quantizer_impl is set to True, the quantization will be performed but the output
        # will be discarded through the hook. It is useful for collecting activation stats,
        # for example during activation calibration in PTQ
        for module in model.modules():
            if isinstance(module, ActQuantProxyFromInjectorBase):
                module.train(is_training)
                if self.call_act_quantizer_impl:
                    for m in module.modules():
                        if hasattr(m, 'observer_only'):
                            m.observer_only = True
                else:
                    module.disable_quant = True
            elif isinstance(module, _ACC_PROXIES):
                module.train(is_training)
                module.disable_quant = True

    def disable_param_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.train(is_training)
                module.disable_quant = True

    def disable_bias_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _BIAS_PROXIES):
                module.train(is_training)
                module.disable_quant = True

    def enable_act_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _ACC_PROXIES):
                module.train(is_training)
                module.disable_quant = False
            elif isinstance(module, ActQuantProxyFromInjectorBase):
                module.disable_quant = False
                module.train(is_training)
                for m in module.modules():
                    if hasattr(m, 'observer_only'):
                        m.observer_only = False

    def enable_param_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.disable_quant = False
                module.train(is_training)

    def enable_bias_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _BIAS_PROXIES):
                module.disable_quant = False
                module.train(is_training)

    def apply(self, model, is_training, quantization_enabled):
        if not quantization_enabled:
            self.disable_act_quantization(model, is_training)
            self.disable_param_quantization(model, is_training)
        else:
            self.enable_act_quantization(model, is_training)
            self.enable_param_quantization(model, is_training)


class _BiasCorrection(DisableEnableQuantization):

    LAYERS = (QuantWBIOL,)

    def __init__(self, layers=LAYERS, skip_if_no_bias=False):
        super(_BiasCorrection, self).__init__()
        self.layers = layers
        self.iterations = {}
        self.correction_map = {}
        self.float_mean_map = {}
        self.collect_float_mean_hooks = []
        self.correct_bias_hooks = []
        self.skip_if_no_bias = skip_if_no_bias

    def compute_mean(self, inp, transpose_dim):
        inp = inp.transpose(0, transpose_dim)
        return inp.reshape(inp.shape[0], -1).mean(dim=1).detach()

    def channel_dim(self, inp, module):
        if len(inp.shape) == 3 and isinstance(module, QuantLinear):
            channel_dim = 2
        else:
            channel_dim = 1
        return channel_dim

    def collect_float_mean(self, module, inp, name):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            raise RuntimeError("Module to bias-correct called multiple times, not supported.")
        transpose_dim = self.channel_dim(inp, module)
        self.float_mean_map[name] = self.compute_mean(inp, transpose_dim)

    def update_correction(self, name, error):
        if name not in self.correction_map:
            self.correction_map[name] = error
        else:
            self.correction_map[name] += error

    def apply_correction(self, model):
        for name, module in model.named_modules():
            if name in self.correction_map.keys():
                correction = self.correction_map[name] / self.iterations[name]
                # When accelerate is enabled, bring tensors onto the device to avoid allocating a meta parameter.
                if hasattr(module, 'allocate_params'):
                    module.allocate_params(module)
                if module.bias is not None:
                    module.bias.data += correction
                elif self.skip_if_no_bias is False:
                    # If accelerate is enabled, bias will be on the same execution device as the weights, but won't be managed properly by accelerate
                    module.register_parameter(
                        'bias', nn.Parameter(correction).to(module.weight.device))
                # Offload params again
                if hasattr(module, 'offload_params'):
                    module.offload_params(module)

    def compute_correct_bias(self, module, inp, name):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            transpose_dim = self.channel_dim(inp, module)
            quant_mean = self.compute_mean(inp, transpose_dim)
            error = self.float_mean_map[name] - quant_mean
            self.update_correction(name, error)
            del self.float_mean_map[name]

    def register_hook_to_wbiol(self, model, hooks):
        """
        Forward hooks are registered to the WBIOL layers.
        In this way, if more hooks are registered to these layers, they will be only called once per module call.
        We perform two more forwards for each WBIOL, but since we call forward and not __call__, eventual hooks would be skipped.
        This is a desired behaviour, since these extra forwards are only necessary to compute the bias correction factor.

        If we registered a single hook for the entire model, and hooks were added to WBIOL layers,
        these would be called by our extra forwards, which would be an unexpected behaviours.
        """
        for name, module in model.named_modules():
            if isinstance(module, self.layers):
                self.iterations[name] = 0
                hook_fn = partial(self.forward_hook_wbiol, name=name)
                hooks.append(module.register_forward_hook(hook_fn))

    def forward_hook_wbiol(self, module, inp, output, name):
        """
        After each forward, we perform two extra forwards to register float and quant outputs
        and the error between the two.

        We do not return the original quant output, but the float one, to avoid error accumulation
        """
        # Compute float reference
        self.disable_act_quantization(module, is_training=False)
        self.disable_param_quantization(module, is_training=False)

        out_float = module.forward(*inp)  # Required to avoid infinite recursion
        self.collect_float_mean(module, out_float, name)
        self.enable_act_quantization(module, is_training=False)
        self.enable_param_quantization(module, is_training=False)
        # Keep output quant disabled until further notice
        self.disable_act_quantization(module.output_quant, is_training=False)
        out_quant = output
        self.compute_correct_bias(module, out_quant, name)
        self.iterations[name] += 1
        return out_float


class norm_correction_mode:

    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            for m in self.model.modules():
                if isinstance(m, BN_LAYERS):
                    m.register_buffer("old_running_mean", m.running_mean.clone())
                    m.register_buffer("old_running_var", m.running_var.clone())
                    m.reset_running_stats()
                    m.momentum = None
                    m.train()

    def __exit__(self, type, value, traceback):
        if self.enabled:
            for m in self.model.modules():
                if isinstance(m, BN_LAYERS):
                    a = torch.sqrt(m.old_running_var / m.running_var)
                    b = m.old_running_mean - a * m.running_mean
                    m.running_var = m.old_running_var.clone() / a
                    m.running_mean = (m.old_running_mean.clone() - b) / a
                    del m.old_running_var
                    del m.old_running_mean
