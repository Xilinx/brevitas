from functools import partial
from abc import ABC

from torch import nn

from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.utils import compute_channel_view_shape
from brevitas.quant_tensor import QuantTensor
from .base import Transform

__all__ = [
    'ClipFloatWeights',
    'DisableEnableQuantization',
    'DisableQuantInference',
    'BiasCorrection'
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
        

class DisableEnableQuantization(Transform, ABC):
    
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
        return QuantTensor(value=inp, training=module.training)
    
    def disable_act_quantization(self, model):
        for module in model.modules():
            if isinstance(module, ActQuantProxyFromInjector):
                hook = module.register_forward_hook(self.disable_act_quant_hook)
                self.disable_act_quant_hooks.append(hook)
            elif isinstance(module, _ACC_PROXIES):
                module.disable_quant = True

    def disable_param_quantization(self, model):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.disable_quant = True
    
    def enable_act_quantization(self, model):
        for module in model.modules():
            if isinstance(module, _ACC_PROXIES):
                module.disable_quant = False
        for hook in self.disable_act_quant_hooks:
            hook.remove()
        self.disable_act_quant_hooks = []

    def enable_param_quantization(self, model):
        for module in model.modules():
            if isinstance(module, _PARAM_PROXIES):
                module.disable_quant = False


class DisableQuantInference(DisableEnableQuantization):

    def apply(self, model, inp):
        self.disable_act_quantization(model)
        self.disable_param_quantization(model)
        output = model(inp)
        self.enable_act_quantization(model)
        self.enable_param_quantization(model)
        return output


class BiasCorrection(DisableEnableQuantization):

    def __init__(self, iterations):
        super(BiasCorrection, self).__init__()
        self.iterations = iterations
        self.curr_iter = 0
        self.correction_map = {}
        self.float_mean_map = {}
        self.collect_float_mean_hooks = []
        self.correct_bias_hooks = []

    def compute_mean(self, inp):
        inp = inp.transpose(0, 1)
        return inp.reshape(inp.shape[0], -1).mean(dim=1).detach()

    def collect_float_mean_hook(self, module, inp, name):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            raise RuntimeError("Module to bias-correct called multiple times, not supported.")
        self.float_mean_map[name] = self.compute_mean(inp)

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
            parent_module.bias = nn.Parameter(correction).to(parent_module.weight.device)

    def correct_bias_hook(self, module, inp, name, parent_module):
        inp = self.unpack_input(inp)
        if name in self.float_mean_map.keys():
            quant_mean = self.compute_mean(inp)
            error = self.float_mean_map[name] - quant_mean
            self.update_correction(name, error)
            if self.curr_iter == self.iterations - 1:
                self.apply_correction(name, parent_module)
            del self.float_mean_map[name]
            inp_broadcast_shape = compute_channel_view_shape(inp, channel_dim=1)
            return inp + error.reshape(inp_broadcast_shape)

    def collect_float_mean(self, model, inp):
        for name, module in model.named_modules():
            if isinstance(module, QuantWBIOL):
                hook_fn = partial(self.collect_float_mean_hook, name=name)
                hook = module.output_quant.register_forward_pre_hook(hook_fn)
                self.collect_float_mean_hooks.append(hook)
        model(inp)
        for hook in self.collect_float_mean_hooks:
            hook.remove()
        self.collect_float_mean_hooks = []

    def correct_bias(self, model, inp):
        for name, module in model.named_modules():
            if isinstance(module, QuantWBIOL):
                hook_fn = partial(self.correct_bias_hook, name=name, parent_module=module)
                hook = module.output_quant.register_forward_pre_hook(hook_fn)
                self.correct_bias_hooks.append(hook)
        model(inp)
        for hook in self.correct_bias_hooks:
            hook.remove()
        self.correct_bias_hooks = []
        assert not self.float_mean_map

    def apply(self, model, inp):
        self.disable_act_quantization(model)
        self.disable_param_quantization(model)
        self.collect_float_mean(model, inp)
        self.enable_act_quantization(model)
        self.enable_param_quantization(model)
        self.correct_bias(model, inp)
        self.curr_iter += 1
        return model
