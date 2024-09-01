# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.config import JIT_ENABLED


def _disabled(fn):
    return fn


if JIT_ENABLED:

    script_method = torch.jit.script_method
    script = torch.jit.script
    ignore = torch.jit.ignore
    ScriptModule = torch.jit.ScriptModule
    Attribute = torch.jit.Attribute

else:

    script_method = _disabled
    script = _disabled
    ignore = _disabled
    ScriptModule = torch.nn.Module
    Attribute = lambda val, type: val
