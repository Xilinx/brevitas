# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import os

from pytest_cases import fixture
from pytest_cases import parametrize

from brevitas.export import export_qonnx
from brevitas.export import export_qonnx_dynamo


@fixture
@parametrize(
    'export_fn', [export_qonnx, partial(export_qonnx_dynamo, dynamo=True, optimize=True)],
    ids=["torchscript", "dynamo"])
def qonnx_export_fn(export_fn):
    yield export_fn


def rm_onnx(path):
    os.remove(path)
    try:
        os.remove(f"{path}.data")
    except OSError:
        pass  # Dynamo-based export will always create this file, Torchscript export will not
