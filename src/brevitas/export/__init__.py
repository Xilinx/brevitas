# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import wraps

from packaging.version import parse

from brevitas import torch_version

from .onnx.debug import enable_debug
from .onnx.qonnx.manager import QONNXDynamoManager
from .onnx.qonnx.manager import QONNXManager
from .onnx.standard.qcdq.manager import StdQCDQONNXDynamoManager
from .onnx.standard.qcdq.manager import StdQCDQONNXManager
from .torch.qcdq.manager import TorchQCDQManager

# The `dynamo` keyword was added to torch.onnx.export in PyTorch 2.5; on older
# versions it must not be forwarded at all (only the TorchScript exporter exists).
_DYNAMO_AVAILABLE = torch_version >= parse("2.5")
# PyTorch's own default for torch.onnx.export flips to the dynamo exporter in 2.9.
_DEFAULT_DYNAMO = torch_version >= parse("2.9")


@wraps(QONNXManager.export)
def export_brevitas_onnx(*args, **kwargs):  # alias for qonnx
    return QONNXManager.export(*args, **kwargs)


def export_qonnx(*args, **kwargs):

    @wraps(QONNXManager.export)
    def _export_qonnx_torchscript(*args, **kwargs):
        return QONNXManager.export(*args, **kwargs)

    @wraps(QONNXDynamoManager.export)
    def _export_qonnx_dynamo(*args, **kwargs):
        return QONNXDynamoManager.export(*args, **kwargs)

    key = "dynamo"
    if _DYNAMO_AVAILABLE:
        kwargs.setdefault(key, _DEFAULT_DYNAMO)
    if kwargs.get(key, False):
        return _export_qonnx_dynamo(*args, **kwargs)
    # TorchScript path: torch < 2.5 has no `dynamo` kwarg, so don't forward it.
    if not _DYNAMO_AVAILABLE:
        kwargs.pop(key, None)
    return _export_qonnx_torchscript(*args, **kwargs)


def export_onnx_qcdq(*args, **kwargs):

    @wraps(StdQCDQONNXManager.export)
    def _export_onnx_qcdq_torchscript(*args, **kwargs):
        return StdQCDQONNXManager.export(*args, **kwargs)

    @wraps(StdQCDQONNXDynamoManager.export)
    def _export_onnx_qcdq_dynamo(*args, **kwargs):
        return StdQCDQONNXDynamoManager.export(*args, **kwargs)

    key = "dynamo"
    if _DYNAMO_AVAILABLE:
        kwargs.setdefault(key, _DEFAULT_DYNAMO)
    if kwargs.get(key, False):
        return _export_onnx_qcdq_dynamo(*args, **kwargs)
    # TorchScript path: torch < 2.5 has no `dynamo` kwarg, so don't forward it.
    if not _DYNAMO_AVAILABLE:
        kwargs.pop(key, None)
    return _export_onnx_qcdq_torchscript(*args, **kwargs)


@wraps(TorchQCDQManager.export)
def export_torch_qcdq(*args, **kwargs):
    return TorchQCDQManager.export(*args, **kwargs)
