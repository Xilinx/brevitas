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

import numpy as np
import torch
import torch.nn.functional as F

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

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
        device = self.layer.weight.device
        model_input = model_input.to(device)

        handle = self.layer.register_forward_hook(self.data_saver)
        self.disable_param_quantization(self.model, is_training=False)
        with torch.no_grad():
            try:
                _ = self.model(model_input)
            except StopFwdException:
                pass
            out = self.data_saver.output_store.detach()

        self.data_saver.store_output = False
        self.enable_param_quantization(self.model, is_training=False)
        with torch.no_grad():
            try:
                _ = self.model(model_input)
            except StopFwdException:
                pass
            inp = self.data_saver.input_store[0].detach()
        handle.remove()
        self.data_saver.store_output = True
        return inp, out


def sigmoid(x):
    return (1.0 + np.exp(-x)) ** -1.0


class TempDecay:

    def __init__(
            self,
            t_max,
            b_range=(20.0, 2.0),
            rel_decay_start=0.0,
            decay_type='cosine',
            decay_shape=1.0):
        self.t_max = t_max
        self.start_b, self.end_b = b_range
        self.decay_type = decay_type
        self.decay_shape = decay_shape
        self.decay_start = rel_decay_start * t_max

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_b

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        if self.decay_type == 'linear':
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
        elif self.decay_type == 'cosine':
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
        elif self.decay_type == 'sigmoid':
            d = self.decay_shape
            offset = sigmoid(-d / 2)
            rel_progress = (sigmoid(d * (rel_t - 0.5)) - offset) / (1 - 2 * offset)
            return self.start_b + (self.end_b - self.start_b) * rel_progress
        elif self.decay_type == 'power':
            return self.end_b + (self.start_b - self.end_b) * (1 - rel_t ** self.decay_shape)
        elif self.decay_type == 'exp':
            r = self.decay_shape
            rel_progress = (1.0 - np.exp(-r * rel_t)) / (1.0 - np.exp(-r))
            return self.start_b + (self.end_b - self.start_b) * rel_progress
        elif self.decay_type == 'log':
            r = self.decay_shape
            C = np.exp(self.end_b / r)
            c = np.exp(self.start_b / r)
            return r * np.log((C - c) * rel_t + c)
        else:
            raise ValueError(f'Unknown temp decay type {self.decay_type}')


class CombinedLoss:

    def __init__(
            self,
            module,
            learned_round_module,
            weight=0.01,
            max_count=1000,
            b_range=(20, 2),
            warmup=0.2,
            decay_start=0.0,
            **temp_decay_kw):
        self.weight = weight
        self.module = module
        self.loss_start = max_count * warmup
        self.temp_decay = TempDecay(
            max_count,
            b_range=b_range,
            rel_decay_start=warmup + (1.0 - warmup) * decay_start,
            **temp_decay_kw,
        )
        self.iter = 0
        self.learned_round_module = learned_round_module

    def __call__(self, pred, tgt, *args, **kwargs):
        self.iter += 1

        rec_loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = 0
        else:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.learned_round_module.p_forward()
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        return total_loss, rec_loss, round_loss


def find_learned_round_module(module):
    for submodule in module.modules():
        if isinstance(submodule, LearnedRoundSte):
            return submodule
    return False


def insert_learned_round_quantizer(layer, learned_round_zeta=1.1, learned_round_gamma=-0.1):
    if isinstance(layer, QuantWBIOL):
        if not find_learned_round_module(layer):
            floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
            delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
            value = -torch.log((learned_round_zeta - learned_round_gamma) /
                               (delta - learned_round_gamma) - 1)
            state_dict = layer.weight_quant.state_dict()
            layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
                float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
                learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID,
                learned_round_gamma=learned_round_gamma,
                learned_round_zeta=learned_round_zeta,
                learned_round_init=value)
            layer.weight_quant.init_tensor_quant()
            layer.weight_quant.load_state_dict(state_dict)


def split_layers(model, blocks):
    for module in model.children():
        if isinstance(module, QuantWBIOL):
            blocks.append(module)
        else:
            split_layers(module, blocks)


def block_wise_learned_round_iterator(model, blocks, iters=1000):
    for block in blocks:
        insert_learned_round_quantizer(block)

        for p in block.parameters():
            p.requires_grad = False

        learned_round_module = find_learned_round_module(block)
        learned_round_module.value.requires_grad = True
        inp_out_class = GetLayerInpOut(model, block)
        layer_loss = CombinedLoss(
            module=block, learned_round_module=learned_round_module, max_count=iters)
        block.train()
        yield block, inp_out_class, layer_loss, learned_round_module
        block.eval()
