from functools import partial
from abc import ABC

from torch import nn

from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn import QuantLinear, QuantHardTanh
from brevitas.nn.utils import compute_channel_view_shape
from brevitas.quant_tensor import QuantTensor
import torch.nn.functional as F
from .base import Transform

__all__ = [
    'ClipFloatWeights',
    'DisableEnableQuantization',
    'BiasCorrection',
    'calibration_mode'
]

_PARAM_PROXIES = (
    WeightQuantProxyFromInjector,
    BiasQuantProxyFromInjector)

_ACC_PROXIES = (
    TruncQuantProxyFromInjector,
    ClampQuantProxyFromInjector)

_LAYERS_TO_CLIP = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d)


def finalize_collect_stats(module):
    if hasattr(module, 'collect_stats_steps') and hasattr(module, 'counter'):
        module.counter = module.collect_stats_steps


class calibration_mode(object):
    def __init__(self, model, enabled=True):
        self.model = model
        self.previous_training_state = model.training
        self.disable_quant_inference = DisableEnableQuantization()
        self.enabled=enabled

    def __enter__(self):
        if self.enabled:
            self.disable_quant_inference.apply(self.model, is_training=True, quantization_enabled=False)

    def __exit__(self, type, value, traceback):
        self.disable_quant_inference.apply(self.model, is_training=self.previous_training_state, quantization_enabled=True)


class ClipFloatWeights(Transform):

    def __init__(self, threshold=15., layers_to_clip=_LAYERS_TO_CLIP) -> None:
        super(ClipFloatWeights, self).__init__()
        self.threshold = threshold
        self.layers_to_clip = layers_to_clip

    def apply(self, model):
        for module in model.modules():
            if isinstance(module, self.layers_to_clip):
                module.weight.data.clamp_(- self.threshold, self.threshold)
        return model
        

class DisableEnableQuantization(Transform):
    
    def __init__(self):
        super(DisableEnableQuantization, self).__init__()
        self.disable_act_quant_hooks = []

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
        for module in model.modules():
            if isinstance(module, ActQuantProxyFromInjector):
                hook = module.register_forward_hook(self.disable_act_quant_hook)
                module.train(is_training)
                self.disable_act_quant_hooks.append(hook)
            elif isinstance(module, _ACC_PROXIES):
                module.train(is_training)
                module.disable_quant = True

    def disable_param_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.train(is_training)
                module.disable_quant = True
    
    def enable_act_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _ACC_PROXIES):
                module.train(is_training)
                module.disable_quant = False
            elif isinstance(module, ActQuantProxyFromInjector):
                module.train(is_training)
        for hook in self.disable_act_quant_hooks:
            hook.remove()
        self.disable_act_quant_hooks = []

    def enable_param_quantization(self, model, is_training):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.disable_quant = False
                module.train(is_training)

    def apply(self, model, is_training, quantization_enabled):
        if not quantization_enabled:
            self.disable_act_quantization(model, is_training)
            self.disable_param_quantization(model, is_training)
        else:
            self.enable_act_quantization(model, is_training)
            self.enable_param_quantization(model, is_training)


class BiasCorrection(DisableEnableQuantization):

    LAYERS = (QuantWBIOL,)

    def __init__(self, iterations, layers=LAYERS):
        super(BiasCorrection, self).__init__()
        self.iterations = iterations
        self.layers = layers
        self.curr_iter = 0
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

    def collect_float_mean_hook(self, module, inp, name, parent_module):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            raise RuntimeError("Module to bias-correct called multiple times, not supported.")
        transpose_dim = self.channel_dim(inp, parent_module)
        self.float_mean_map[name] = self.compute_mean(inp, transpose_dim)

    def update_correction(self, name, error):
        if name not in self.correction_map:
            self.correction_map[name] = error
        else:
            self.correction_map[name] += error

    def apply_correction(self, name, parent_module):
        correction = self.correction_map[name] / self.iterations
        if parent_module.bias is not None:
            parent_module.bias.data += correction
        else:
            parent_module.register_parameter('bias', nn.Parameter(correction).to(parent_module.weight.device))

    def correct_bias_hook(self, module, inp, name, parent_module):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            transpose_dim = self.channel_dim(inp, parent_module)
            quant_mean = self.compute_mean(inp, transpose_dim)
            error = self.float_mean_map[name] - quant_mean
            self.update_correction(name, error)
            if self.curr_iter == self.iterations - 1:
                self.apply_correction(name, parent_module)
            del self.float_mean_map[name]
            inp_broadcast_shape = compute_channel_view_shape(inp, channel_dim=self.channel_dim(inp, parent_module))
            return inp + error.reshape(inp_broadcast_shape)

    def collect_float_mean(self, model, *args, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, self.layers):
                hook_fn = partial(self.collect_float_mean_hook, name=name, parent_module=module)
                hook = module.output_quant.register_forward_pre_hook(hook_fn)
                self.collect_float_mean_hooks.append(hook)
        model(*args, **kwargs)
        for hook in self.collect_float_mean_hooks:
            hook.remove()
        self.collect_float_mean_hooks = []

    def correct_bias(self, model, *args, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, self.layers):
                hook_fn = partial(self.correct_bias_hook, name=name, parent_module=module)
                hook = module.output_quant.register_forward_pre_hook(hook_fn)
                self.correct_bias_hooks.append(hook)
        model(*args, **kwargs)
        for hook in self.correct_bias_hooks:
            hook.remove()
        self.correct_bias_hooks = []
        assert not self.float_mean_map

    def apply(self, model, *model_args, **model_kwargs):
        self.disable_act_quantization(model, is_training=False)
        self.disable_param_quantization(model, is_training=False)
        self.collect_float_mean(model, *model_args, **model_kwargs)
        self.enable_act_quantization(model, is_training=False)
        self.enable_param_quantization(model, is_training=False)
        self.correct_bias(model, *model_args, **model_kwargs)
        self.curr_iter += 1
        return model
