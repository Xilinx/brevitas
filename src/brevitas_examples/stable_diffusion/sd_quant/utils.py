"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

from contextlib import contextmanager

import torch

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas_examples.stable_diffusion.sd_quant.constants import SDXL_FEAT_DIM


class InferenceWeightProxyHandler(BaseHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super(InferenceWeightProxyHandler, self).__init__()
        self.scale = None
        self.zero_point = None
        self.bit_width = None
        self.dtype = None
        self.float_weight = None

    def scaling_impl(self, proxy_module):
        return proxy_module.tensor_quant.scaling_impl

    def zero_point_impl(self, proxy_module):
        return proxy_module.tensor_quant.zero_point_impl

    def bit_width_impl(self, proxy_module):
        return proxy_module.tensor_quant.msb_clamp_bit_width_impl

    def export_scale(self, proxy_module, bit_width):
        scaling_impl = self.scaling_impl(proxy_module)
        int_scaling_impl = proxy_module.tensor_quant.int_scaling_impl
        int_threshold = int_scaling_impl(bit_width)
        threshold = scaling_impl.stats_scaling_impl(scaling_impl.parameter_list_stats())
        return threshold / int_threshold

    def export_zero_point(self, proxy_module, weight, scale, bit_width):
        zero_point_impl = self.zero_point_impl(proxy_module)
        return zero_point_impl(weight, scale, bit_width)

    def prepare_for_export(self, module):
        assert len(module.tracked_module_list) == 1, "Shared quantizers not supported."
        self.bit_width = self.bit_width_impl(module)()
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        quant_layer = module.tracked_module_list[0]
        self.float_weight = quant_layer.quant_weight()
        self.dtype = self.float_weight.value.dtype
        # if (self.float_weight.zero_point != 0.).any():
        #     self.zero_point = self.export_zero_point(module, quant_layer.weight, self.scale, self.bit_width).detach().cpu()
        # self.scale = self.export_scale(module, self.bit_width).detach().cpu()
        # quant_layer.weight.data = quant_layer.weight.data.cpu()

    def forward(self, x):

        return self.float_weight.value, self.float_weight.scale, self.float_weight.zero_point, self.bit_width


class InferenceWeightProxyManager(BaseManager):
    handlers = [InferenceWeightProxyHandler]

    @classmethod
    def set_export_handler(cls, module):
        if hasattr(module,
                   'requires_export_handler') and module.requires_export_handler and not isinstance(
                       module, (WeightQuantProxyFromInjector)):
            return
        _set_proxy_export_handler(cls, module)


def store_mapping_tensor_state_dict(model):
    mapping = dict()
    for module in model.modules():
        if isinstance(module, QuantWeightBiasInputOutputLayer):
            mapping[module.weight.data_ptr()] = module.weight.device
    return mapping


def restore_mapping(model, mapping):
    for module in model.modules():
        if isinstance(module, QuantWeightBiasInputOutputLayer):
            module.weight.data = module.weight.data.to(mapping[module.weight.data_ptr()])


@contextmanager
def brevitas_proxy_inference_mode(model):
    mapping = store_mapping_tensor_state_dict(model)
    is_training = model.training
    model.eval()
    model.apply(InferenceWeightProxyManager.set_export_handler)
    _set_proxy_export_mode(model, enabled=True, proxy_class=WeightQuantProxyFromInjector)
    try:
        yield model
    finally:
        restore_mapping(model, mapping)
        _set_proxy_export_mode(model, enabled=False)
        model.train(is_training)


def unet_input_shape(resolution):
    return (4, resolution // 8, resolution // 8)


def generate_latents(seeds, device, dtype, input_shape):
    """
    Generate a concatenation of latents of a given input_shape
    (batch size excluded) on a target device from one or more seeds.
    """
    latents = None
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator = generator.manual_seed(seed)
        image_latents = torch.randn((1, *input_shape),
                                    generator=generator,
                                    device=device,
                                    dtype=dtype)
        latents = image_latents if latents is None else torch.cat((latents, image_latents))
    return latents


def generate_unet_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    sample = torch.randn(batch_size, *unet_input_shape, device=device, dtype=dtype)
    unet_rand_inputs = {
        'sample':
            sample,
        'timestep':
            torch.tensor(1, dtype=torch.int64, device=device),
        'encoder_hidden_states':
            torch.randn(batch_size, *embedding_shape, device=device, dtype=dtype)}
    if with_return_dict_false:
        unet_rand_inputs['return_dict'] = False
    return unet_rand_inputs


def generate_unet_21_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    unet_rand_inputs = generate_unet_rand_inputs(
        embedding_shape, unet_input_shape, batch_size, device, dtype, with_return_dict_false)
    return tuple(unet_rand_inputs.values())


def generate_unet_xl_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    # We need to pass a combination of args and kwargs to ONNX export
    # If we pass all kwargs, something breaks
    # If we pass only the last element as kwargs, since it is a dict, it has a weird interaction and something breaks
    # The solution is to pass only one argument as args, and everything else as kwargs
    unet_rand_inputs = generate_unet_rand_inputs(
        embedding_shape, unet_input_shape, batch_size, device, dtype, with_return_dict_false)
    sample = unet_rand_inputs['sample']
    del unet_rand_inputs['sample']
    unet_rand_inputs['timestep_cond'] = None
    unet_rand_inputs['cross_attention_kwargs'] = None
    unet_rand_inputs['added_cond_kwargs'] = {
        "text_embeds": torch.randn(1, SDXL_FEAT_DIM, dtype=dtype, device=device),
        "time_ids": torch.randn(1, 6, dtype=dtype, device=device)}
    inputs = (sample, unet_rand_inputs)
    return inputs
