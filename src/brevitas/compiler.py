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

# PyTorch 2.3.1 can incorrectly eliminate small helper functions during compile.
# Newer versions can trace these helpers directly.
if version.parse('2.1') < torch_version < version.parse('2.4'):
    disable_on_old_torch = torch._dynamo.disable
else:
    disable_on_old_torch = _disabled
