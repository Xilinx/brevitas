# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from packaging import version

from brevitas import torch_version


def onnx_export_opset():
    try:
        import torch.onnx.symbolic_helper as cfg
        ATR_NAME = '_export_onnx_opset_version'
        opset = getattr(cfg, ATR_NAME)

    except:
        if torch_version < version.parse('2.9.0'):
            from torch.onnx._globals import GLOBALS as cfg
        else:
            from torch.onnx._internal.torchscript_exporter._globals import GLOBALS as cfg

        ATR_NAME = 'export_onnx_opset_version'
        opset = getattr(cfg, ATR_NAME)

    return opset
