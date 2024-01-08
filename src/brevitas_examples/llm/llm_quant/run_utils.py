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
import functools

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTModel
from transformers.utils.fx import symbolic_trace

from brevitas.fx.brevitas_tracer import value_trace
from brevitas.fx.value_tracer import ValueProxy
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.utils.python_utils import recurse_getattr

BLOCK_PATTERNS = [
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",]


def get_fx_graph(model, ref_kwargs=None, dtype=None):
    try:
        graph_model = symbolic_trace(model, list(ref_kwargs.keys()))
    except:
        assert ref_kwargs is not None, "Symbolic traced failed, pass an example input to perform FX value trace "
        with cast_to_float32(model, dtype):
            graph_model = value_trace(model, value_args=ref_kwargs)

    graph_model = TorchFunctionalToModule().apply(graph_model)
    return graph_model


def get_preceding_modules(model: nn.Module, module_name: str):
    # From https://github.com/huggingface/optimum/blob/main/optimum/gptq/utils.py
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model: nn.Module, module_name: str, name: str = ""):
        nonlocal stop_adding
        for name_bis, child in model.named_children():
            new_name = name + "." + name_bis if name != "" else name_bis
            if new_name == module_name:
                stop_adding = True
                break
            _get_preceding_modules(child, module_name, name=new_name)
        if not stop_adding:
            previous_module_name.append(name)
        return previous_module_name

    return _get_preceding_modules(model, module_name)


def get_block_name_with_pattern(model: nn.Module):
    """
    From: https://github.com/huggingface/optimum/blob/main/optimum/gptq/utils.py
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(pattern_candidate in name for name in modules_names):
            return pattern_candidate
    raise ValueError(
        "Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`"
    )


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
def capture_first_layer_inputs(input_capture_fn, dataloader, model, layers, preceding_layers_name):

    for module_name in preceding_layers_name:
        module = recurse_getattr(model, module_name)
        if module is None:
            raise ValueError(f"Module {module_name} was not found in model")
        module = module.cuda()

    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0}

    class InputCatcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs.keys():
                cache['position_ids'] = kwargs['position_ids']
            raise InputCatcherException

    layers[0] = InputCatcher(layers[0])
    input_capture_fn(model, dataloader)
    inps = torch.cat(inps, dim=0).cuda().to(dtype)

    layers[0] = layers[0].module

    for module_name in preceding_layers_name:
        module = recurse_getattr(model, module_name)
        if module is None:
            raise ValueError(f"Module {module_name} was not found in model")
        module = module.cpu()
    return inps, cache


@torch.no_grad()
def apply_layer_inference_fn(
        model, dataloader, inference_fn, input_capture_fn, block_name=None, **inference_fn_kwargs):
    if block_name is None:
        block_name = get_block_name_with_pattern(model)

    layers = recurse_getattr(model, block_name)
    module_name_preceding_first_block = get_preceding_modules(model, block_name)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, cache = capture_first_layer_inputs(
        input_capture_fn, dataloader, model, layers, module_name_preceding_first_block)
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


def apply_layer_ptq_fn(model, dataloader, inference_fn, **inference_fn_kwargs):
    return apply_layer_inference_fn(
        model,
        dataloader,
        inference_fn,
        input_capture_fn=calib_input_capture,
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
