# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def clear_class_registry():
    # torch.jit.trace leaks memory, this should help
    # https://github.com/pytorch/pytorch/issues/86537
    # https://github.com/pytorch/pytorch/issues/35600
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._script_classes.clear()
