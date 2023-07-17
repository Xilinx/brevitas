"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Adapted from https://github.com/IST-DASLab/gptq, released under the following LICENSE:

Copyright 2023 IST-DASLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTModel

from brevitas.fx.value_tracer import ValueProxy


def get_model_impl(model):
    model_impl = model.model
    if isinstance(model_impl, OPTModel):
        model_impl = model_impl.decoder
    return model_impl


class InputCatcherException(Exception):
    pass


@torch.no_grad()
def calib_input_capture(model, dataloader):
    for batch in dataloader:
        batch = batch.cuda()
        try:
            model(batch)
        except InputCatcherException:
            pass


@torch.no_grad()
def capture_first_layer_inputs(input_capture_fn, dataloader, model, model_impl, nsamples, seqlen):
    layers = model_impl.layers

    model_impl.embed_tokens = model_impl.embed_tokens.cuda()
    if hasattr(model_impl, 'embed_positions'):
        model_impl.embed_positions = model_impl.embed_positions.cuda()
    if hasattr(model_impl, 'project_in') and model_impl.project_in is not None:
        model_impl.project_in = model_impl.project_in.cuda()
    if hasattr(model_impl, 'norm'):
        model_impl.norm = model_impl.norm.cuda()
    if hasattr(model_impl, 'embed_layer_norm'):
        model_impl.embed_layer_norm = model_impl.embed_layer_norm.cuda()

    layers[0] = layers[0].cuda()

    dtype = next(iter(model_impl.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype).cuda()
    cache = {'i': 0}

    class InputCatcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs.keys():
                cache['position_ids'] = kwargs['position_ids']
            raise InputCatcherException

    layers[0] = InputCatcher(layers[0])
    input_capture_fn(model, dataloader)

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model_impl.embed_tokens = model_impl.embed_tokens.cpu()
    if hasattr(model_impl, 'embed_positions'):
        model_impl.embed_positions = model_impl.embed_positions.cpu()
    if hasattr(model_impl, 'project_in') and model_impl.project_in is not None:
        model_impl.project_in = model_impl.project_in.cpu()
    if hasattr(model_impl, 'norm'):
        model_impl.norm = model_impl.norm.cpu()
    if hasattr(model_impl, 'embed_layer_norm'):
        model_impl.embed_layer_norm = model_impl.embed_layer_norm.cpu()

    return inps, cache


@torch.no_grad()
def apply_layer_inference_fn(
        model,
        dataloader,
        nsamples,
        inference_fn,
        input_capture_fn,
        seqlen=2048,
        **inference_fn_kwargs):
    model_impl = get_model_impl(model)
    layers = model_impl.layers

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, cache = capture_first_layer_inputs(
        input_capture_fn, dataloader, model, model_impl, nsamples, seqlen)
    outs = torch.zeros_like(inps)

    cached_values = {}
    cached_values['attention_mask'] = cache['attention_mask']
    if 'position_ids' in cache.keys():
        cached_values['position_ids'] = cache['position_ids']

    for curr_layer in tqdm(layers):
        inference_fn(curr_layer, inps, outs, cached_values, **inference_fn_kwargs)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return inps


def apply_layer_ptq_fn(
        model, dataloader, nsamples, inference_fn, seqlen=2048, **inference_fn_kwargs):
    return apply_layer_inference_fn(
        model,
        dataloader,
        nsamples,
        inference_fn,
        input_capture_fn=calib_input_capture,
        seqlen=seqlen,
        **inference_fn_kwargs)


@contextmanager
def cast_to_float32(model, target_dtype):
    dtype_dict = {}
    for name, p in model.state_dict().items():
        # This allows to pick up duplicated parameters
        dtype_dict[name] = p.dtype
    if any(dtype != torch.float32 for dtype in dtype_dict.values()):
        model.to(dtype=torch.float32)
    try:
        yield model
    finally:
        for name, p in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items():
            if name in dtype_dict:
                p.data = p.data.to(dtype_dict[name])
            else:
                # target_dtype covers any new tensors that might have been
                # introduced in the process (e.g. during equalization)
                p.data = p.data.to(target_dtype)


class CastFloat16ToFloat32(TorchDispatchMode):

    def cast_cpu_to(self, x, src_dtype, dest_dtype):
        # workaround for value_trace to avoid tracing through the ops below
        if issubclass(type(x), ValueProxy):
            t = x.value
        else:
            t = x
        if isinstance(t, torch.Tensor) and t.dtype == src_dtype and t.device == torch.device('cpu'):
            # Keep the casting out of place so that it's ephemeral
            return t.to(dest_dtype, non_blocking=True, copy=True)
        return x

    def cast_cpu_float32(self, t):
        return self.cast_cpu_to(t, src_dtype=torch.float16, dest_dtype=torch.float32)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        args, kwargs = tree_map(self.cast_cpu_float32, (args, kwargs))
        out = func(*args, **kwargs)
        return out
