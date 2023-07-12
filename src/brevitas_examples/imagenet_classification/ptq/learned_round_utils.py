# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Part of this code has been re-adapted from https://github.com/yhhhli/BRECQ
# under the following LICENSE:

# MIT License

# Copyright (c) 2021 Yuhang Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy
import re

import torch

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector

config.IGNORE_MISSING_KEYS = True


class StopFwdException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


class DataSaverHook:

    def __init__(self, store_output: False):
        self.store_output = store_output
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_output:
            self.output_store = output_batch
        self.input_store = input_batch
        raise StopFwdException


class GetLayerInpOut(DisableEnableQuantization):

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module, store_output: bool = True):
        super().__init__()
        self.model = model
        self.layer = layer
        self.store_output = store_output
        self.data_saver = DataSaverHook(store_output=store_output)

    def __call__(self, model_input):
        device = next(self.layer.parameters()).device
        model_input = model_input.to(device)
        self.model.eval()
        handle = self.layer.register_forward_hook(self.data_saver)
        self.disable_param_quantization(self.model, is_training=False)
        self.disable_act_quantization(self.model, is_training=False)
        with torch.no_grad():
            try:
                _ = self.model(model_input)
            except StopFwdException:
                pass
            out = self.data_saver.output_store.detach()

        self.data_saver.store_output = False
        self.enable_param_quantization(self.model, is_training=False)
        self.enable_act_quantization(self.model, is_training=False)
        with torch.no_grad():
            try:
                _ = self.model(model_input)
            except StopFwdException:
                pass
            inp = self.data_saver.input_store[0].detach()
        handle.remove()
        self.data_saver.store_output = True
        return inp, out


class LinearTempDecay:

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class Loss:

    def __init__(
            self,
            module,
            learned_round_modules,
            rec_loss='mse',
            weight=0.01,
            max_count=1000,
            b_range=(20, 2),
            warmup=0.2,
            decay_start=0.0):
        self.weight = weight
        self.module = module
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(
            max_count, start_b=b_range[0], end_b=b_range[1], rel_start_decay=warmup)
        self.iter = 0
        self.learned_round_modules = learned_round_modules
        self.rec_loss = rec_loss

    def __call__(self, pred, tgt):
        self.iter += 1

        if self.rec_loss == 'mse':
            rec_loss = (pred - tgt).abs().pow(2.).sum(1).mean()
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = 0
        else:  # 1 - |(h-0.5)*2|**b
            round_loss = 0
            b = self.temp_decay(self.iter)
            for learned_round_module in self.learned_round_modules:
                round_vals = learned_round_module.p_forward()
                round_loss += self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        return total_loss, rec_loss, round_loss, b


def find_learned_round_module(module):
    for submodule in module.modules():
        if isinstance(submodule, LearnedRoundSte):
            return submodule
    return False


def insert_learned_round_quantizer(layer, learned_round_zeta=1.1, learned_round_gamma=-0.1):
    if not find_learned_round_module(layer):
        floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
        delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
        value = -torch.log((learned_round_zeta - learned_round_gamma) /
                           (delta - learned_round_gamma) - 1)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            bit_width=4,
            float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
            learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID,
            learned_round_gamma=learned_round_gamma,
            learned_round_zeta=learned_round_zeta,
            learned_round_init=value)
        layer.weight_quant.init_tensor_quant(preserve_state_dict=True)


def split_layerwise(model, class_type=QuantWBIOL):
    layers = dict()
    for name, module in model.named_modules():
        if isinstance(module, class_type):
            layers[name] = module
    return layers


def split_blockwise(model, block_name):
    regex = re.compile(block_name)
    blocks = dict()
    for name, module in model.named_modules():
        regex_match = regex.match(name)
        if regex_match:
            blocks[regex_match.string] = module
    return blocks


def layerwise_learned_round_iterator(layers, iters=1000, learned_round_type='layerwise'):
    for name, layers in layers.items():
        for p in layers.parameters():
            p.requires_grad = False

        learned_round_modules = []
        scale_parameters = []
        for module in layers.modules():
            if isinstance(module, QuantWBIOL):
                insert_learned_round_quantizer(module)
                module = find_learned_round_module(module)
                module.value.requires_grad = True
                learned_round_modules.append(module)
            if isinstance(module, ActQuantProxyFromInjector) and learned_round_type == 'block':
                for p in module.parameters():
                    p.requires_grad = True
                    scale_parameters.append(p)

        layer_loss = Loss(
            module=layers, learned_round_modules=learned_round_modules, max_count=iters)
        yield layers, layer_loss, learned_round_modules, scale_parameters

        layers.eval()


def save_inp_out_data(
        model,
        module,
        dataloader: torch.utils.data.DataLoader,
        store_inp=False,
        store_out=False,
        keep_gpu: bool = True,
        disable_quant=False):
    if disable_quant:
        disable_quant_class = DisableEnableQuantization()
        disable_quant_class.disable_act_quantization(model, False)
        disable_quant_class.disable_param_quantization(model, False)
    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_output=store_out)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for img, t in dataloader:
            try:
                _ = model(img.to(device))
            except StopFwdException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    cached[0].append(data_saver.input_store[0].detach().cpu())
            if store_out:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    if store_inp:
        cached[0] = torch.cat([x for x in cached[0]])
    if store_out:
        cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    if disable_quant:
        disable_quant_class.enable_act_quantization(model, False)
        disable_quant_class.enable_param_quantization(model, False)
    return cached
