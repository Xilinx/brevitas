# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from torch.nn import Module

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.function import LSTMCellFn

from .handler import (
    StdQCDQONNXWeightQuantProxyHandler,
    StdQCDQONNXBiasQuantProxyHandler,
    StdQCDQONNXActQuantProxyHandler,
    StdQCDQONNXDecoupledWeightQuantProxyHandler,
    StdQCDQONNXTruncQuantProxyHandler,
    StdQCDQONNXQuantLSTMLayerHandler)

from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn
from ..manager import StdONNXBaseManager


class StdQCDQONNXManager(StdONNXBaseManager):
    target_name = 'StdQCDQONNX'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        StdQCDQONNXWeightQuantProxyHandler,
        StdQCDQONNXBiasQuantProxyHandler,
        StdQCDQONNXActQuantProxyHandler,
        StdQCDQONNXDecoupledWeightQuantProxyHandler,
        StdQCDQONNXTruncQuantProxyHandler,
        StdQCDQONNXQuantLSTMLayerHandler
    ]

    custom_fns = [
        DebugMarkerFunction,
        QuantizeLinearFn,
        DequantizeLinearFn,
        IntClipFn,
        LSTMCellFn,
    ]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)