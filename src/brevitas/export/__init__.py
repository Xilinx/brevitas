# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging.version import parse

from brevitas import torch_version

# ONNX-related managers are imported lazily so that importing brevitas.export (e.g. for
# inference mode) does not require onnx.

# PyTorch's own default for torch.onnx.export flips to the dynamo exporter in 2.9.
_DEFAULT_DYNAMO = torch_version >= parse("2.9")


def export_brevitas_onnx(*args, **kwargs):  # alias for qonnx
    from .onnx.qonnx.manager import QONNXManager
    return QONNXManager.export(*args, **kwargs)


def export_qonnx(*args, **kwargs):
    from .onnx.qonnx.manager import QONNXDynamoManager
    from .onnx.qonnx.manager import QONNXManager

    key = "dynamo"
    kwargs.setdefault(key, _DEFAULT_DYNAMO)
    if kwargs.get(key, False):
        return QONNXDynamoManager.export(*args, **kwargs)
    return QONNXManager.export(*args, **kwargs)


def export_onnx_qcdq(*args, **kwargs):
    from .onnx.standard.qcdq.manager import StdQCDQONNXDynamoManager
    from .onnx.standard.qcdq.manager import StdQCDQONNXManager

    key = "dynamo"
    kwargs.setdefault(key, _DEFAULT_DYNAMO)
    if kwargs.get(key, False):
        return StdQCDQONNXDynamoManager.export(*args, **kwargs)
    return StdQCDQONNXManager.export(*args, **kwargs)


def export_torch_qcdq(*args, **kwargs):
    from .torch.qcdq.manager import TorchQCDQManager
    return TorchQCDQManager.export(*args, **kwargs)
