# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import torch

from brevitas import torch_version


def _disabled(fn):
    return fn


if torch_version > version.parse('2.1'):
    disable = torch._dynamo.disable
else:
    disable = _disabled
