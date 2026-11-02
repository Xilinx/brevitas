# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import torch


def disable(*args, **kwargs):
    warnings.warn(
        "brevitas.compiler.disable might be deprecated in a future release in favour of "
        "torch.compiler.disable.",
        FutureWarning,
    )
    return torch._dynamo.disable(*args, **kwargs)

