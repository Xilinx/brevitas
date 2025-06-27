# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import os

from packaging.version import parse
from pytest_cases import fixture
from pytest_cases import parametrize

from brevitas import torch_version
from brevitas.export import export_qonnx


def _get_qonnx_export_modes():
    if parse("2.4") > torch_version:
        export_fns = [export_qonnx]
        export_ids = ["torchscript"]
    else:
        export_fns = [export_qonnx, partial(export_qonnx, dynamo=True, optimize=True)]
        export_ids = ["torchscript", "dynamo"]
    return export_fns, export_ids


_qonnx_export_fns, _qonnx_export_ids = _get_qonnx_export_modes()


@fixture
@parametrize('export_fn', _qonnx_export_fns, ids=_qonnx_export_ids)
def qonnx_export_fn(export_fn):
    yield export_fn


def rm_onnx(path):
    os.remove(path)
    try:
        os.remove(f"{path}.data")
    except OSError:
        pass  # Dynamo-based export will always create this file, Torchscript export will not
