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
import inspect

from optimum.utils.normalized_config import NormalizedConfigManager
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from transformers import AutoConfig
from transformers.utils.fx import symbolic_trace

from brevitas.fx.value_tracer import ValueProxy


def get_fx(model):
    forward_signature = inspect.signature(model.forward).parameters
    if all(input_name in forward_signature
           for input_name in ["input_ids", "attention_mask", "past_key_values"]):
        input_names = ["input_ids", "attention_mask", "past_key_values"]
    else:
        raise ValueError(
            f"Quantization with an FX graph is currently only supported for models taking `input_ids`, `attention_mask` and `past_key_values` as inputs. The model only has the following inputs: {forward_signature}"
        )

    with torch.no_grad():
        model = symbolic_trace(model, input_names)
    return model


def modify_dataloader(model_name_or_path, data, dtype):
    config = AutoConfig.from_pretrained(model_name_or_path)

    normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
    normalized_config = normalized_config_class(config)

    num_heads = normalized_config.num_attention_heads
    head_dim = normalized_config.hidden_size // num_heads
    num_layers = normalized_config.num_layers

    for sample in data:
        sample["past_key_values"] = tuple((
            torch.zeros(1, num_heads, 0, head_dim, device=sample["input_ids"].device, dtype=dtype),
            torch.zeros(1, num_heads, 0, head_dim, device=sample["input_ids"].device, dtype=dtype),
        ) for _ in range(num_layers))
    return data


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
