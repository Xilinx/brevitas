# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import torch

from brevitas.config import JIT_ENABLED

# IS_ABOVE_110 = version.parse(torch.__version__) > version.parse('1.1.0')


def _disabled(fn):
    return fn


if JIT_ENABLED:

    script_method = torch.jit.script_method
    script = torch.jit.script
    ScriptModule = torch.jit.ScriptModule
    Attribute = torch.jit.Attribute

    script_method_110_disabled = script_method
    # script_method_110_disabled = _disabled
    # if not IS_ABOVE_110:
    # else:

else:

    script_method = _disabled
    script = _disabled
    script_method_110_disabled = _disabled
    ScriptModule = torch.nn.Module
    Attribute = lambda val, type: val
