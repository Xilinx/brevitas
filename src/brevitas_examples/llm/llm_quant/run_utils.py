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
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from brevitas.fx.value_tracer import ValueProxy
from brevitas.utils.torch_utils import same_storage


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


# This functions remap rewriters so match modules in a potentially different model that shares the same underlying tensors
# We rely on the fact that two versions of the same model (eager vs FX) might have different modules id (id(fx_module) != id (eager_module))
# However, the underlying tensors are still shared, so we can recostruct the mapping between the two
# modules.
# NOTE: two versions of the same model might also wrap the shared tensors in distinct Parameter
# objects (id(fx_param) != id(eager_param)), so we match on shared storage rather than on id.
# Tensors on the meta device (e.g. offloaded by accelerate) all have data_ptr() == 0, so they
# cannot be matched by storage: the remapping must be done before offloading.
def fix_rewriter(rewriters, old_model_ref, tensor_name):
    for r in rewriters:
        old_tensor = getattr(r.old_module_instance, tensor_name)
        if old_tensor.is_meta:
            raise RuntimeError(
                f"Cannot remap a rewriter whose '{tensor_name}' is on the meta device (offloaded): "
                "call fix_rewriter before offloading the model")
        module = []
        for m in old_model_ref.modules():
            tensor = getattr(m, tensor_name, None)
            if torch.is_tensor(tensor) and not tensor.is_meta and same_storage(tensor, old_tensor):
                module.append(m)
        r.old_module_instance = module[0]
    return rewriters
