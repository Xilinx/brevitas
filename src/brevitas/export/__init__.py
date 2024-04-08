# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import wraps

from .onnx.debug import enable_debug
from .onnx.qonnx.manager import QONNXManager
from .onnx.standard.qcdq.manager import StdQCDQONNXManager
from .torch.qcdq.manager import TorchQCDQManager


@wraps(QONNXManager.export)
def export_brevitas_onnx(*args, **kwargs):  # alias for qonnx
    return QONNXManager.export(*args, **kwargs)


@wraps(QONNXManager.export)
def export_qonnx(*args, **kwargs):
    return QONNXManager.export(*args, **kwargs)


@wraps(StdQCDQONNXManager.export)
def export_onnx_qcdq(*args, **kwargs):
    return StdQCDQONNXManager.export(*args, **kwargs)


@wraps(TorchQCDQManager.export)
def export_torch_qcdq(*args, **kwargs):
    return TorchQCDQManager.export(*args, **kwargs)
