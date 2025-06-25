# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

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
