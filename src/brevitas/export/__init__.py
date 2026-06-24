# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import wraps

from .onnx.debug import enable_debug
from .onnx.finn.manager import FINNONNXDynamoManager
from .onnx.finn.manager import FINNONNXManager
from .onnx.qonnx.manager import QONNXDynamoManager
from .onnx.qonnx.manager import QONNXManager
from .onnx.standard.qcdq.manager import StdQCDQONNXManager
from .torch.qcdq.manager import TorchQCDQManager


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
    if kwargs.get(key, False):
        return _export_qonnx_dynamo(*args, **kwargs)
    return _export_qonnx_torchscript(*args, **kwargs)


def export_finn_onnx(*args, **kwargs):

    @wraps(FINNONNXManager.export)
    def _export_finn_onnx_torchscript(*args, **kwargs):
        return FINNONNXManager.export(*args, **kwargs)

    @wraps(FINNONNXDynamoManager.export)
    def _export_finn_onnx_dynamo(*args, **kwargs):
        return FINNONNXDynamoManager.export(*args, **kwargs)

    key = "dynamo"
    if kwargs.get(key, False):
        return _export_finn_onnx_dynamo(*args, **kwargs)
    return _export_finn_onnx_torchscript(*args, **kwargs)


@wraps(StdQCDQONNXManager.export)
def export_onnx_qcdq(*args, **kwargs):
    return StdQCDQONNXManager.export(*args, **kwargs)


@wraps(TorchQCDQManager.export)
def export_torch_qcdq(*args, **kwargs):
    return TorchQCDQManager.export(*args, **kwargs)
