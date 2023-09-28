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
from brevitas.nn.utils import compute_channel_view_shape
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor

from .base import Transform

__all__ = [
    'ClipFloatWeights', 'DisableEnableQuantization', 'bias_correction_mode', 'calibration_mode']

_PARAM_PROXIES = (WeightQuantProxyFromInjector, BiasQuantProxyFromInjector)

_BIAS_PROXIES = (BiasQuantProxyFromInjector)

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

    def __enter__(self):
        if self.enabled:
            self.model.apply(extend_collect_stats_steps)
            self.model.apply(set_collect_stats_to_average)
            self.disable_quant_inference.apply(
                self.model, is_training=True, quantization_enabled=False)

    def __exit__(self, type, value, traceback):
        self.model.apply(finalize_collect_stats)
        self.disable_quant_inference.apply(
            self.model, is_training=self.previous_training_state, quantization_enabled=True)


class bias_correction_mode:

    def __init__(self, model, enabled=True):
        self.model = model
        self.bias_correction = _BiasCorrection()
        self.enabled = enabled
        self.hooks = []

    def __enter__(self):
        if self.enabled:
            self.bias_correction.register_hook_to_wbiol(self.model, self.hooks)

    def __exit__(self, type, value, traceback):
        self.bias_correction.apply_correction(self.model)
        for hook in self.hooks:
            hook.remove()


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
        return QuantTensor(value=inp, training=module.training)

    def disable_act_quantization(self, model, is_training):
        # If self.call_act_quantizer_impl is set to True, the quantization will be performed but the output
        # will be discarded through the hook. It is useful for collecting activation stats,
        # for example during activation calibration in PTQ
        for module in model.modules():
            if isinstance(module, ActQuantProxyFromInjector):
                module.train(is_training)
                if self.call_act_quantizer_impl:
                    hook = module.register_forward_hook(self.disable_act_quant_hook)
                    self.disable_act_quant_hooks.append(hook)
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
            elif isinstance(module, ActQuantProxyFromInjector):
                module.disable_quant = False
                module.train(is_training)
        for hook in self.disable_act_quant_hooks:
            hook.remove()
        self.disable_act_quant_hooks = []

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

    def __init__(self, layers=LAYERS):
        super(_BiasCorrection, self).__init__()
        self.layers = layers
        self.iterations = {}
        self.correction_map = {}
        self.float_mean_map = {}
        self.collect_float_mean_hooks = []
        self.correct_bias_hooks = []

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
                if module.bias is not None:
                    module.bias.data += correction
                else:
                    module.register_parameter(
                        'bias', nn.Parameter(correction).to(module.weight.device))

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

        # Compute quant output
        # We need to disable output_quant while out_quant is being computed
        # or we are going to apply bias correction on post quant values instead of pre quant
        self.disable_act_quantization(module.output_quant, is_training=False)
        out_quant = module.forward(*inp)  # Required to avoid infinite recursion
        self.compute_correct_bias(module, out_quant, name)
        self.enable_act_quantization(module.output_quant, is_training=False)
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
