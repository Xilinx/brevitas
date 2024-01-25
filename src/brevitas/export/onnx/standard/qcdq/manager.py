# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.nn import Module

from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.function import LSTMCellFn

from ..function import DequantizeLinearFn
from ..function import DynamicQuantizeLinearFn
from ..function import IntClipFn
from ..function import QuantizeLinearFn
from ..manager import StdONNXBaseManager
from .handler import StdCDQCastONNXBiasQuantProxyHandler
from .handler import StdCDQCastONNXDecoupledWeightQuantProxyHandler
from .handler import StdCDQCastONNXDecoupledWeightQuantWithInputProxyHandler
from .handler import StdCDQCastONNXWeightQuantProxyHandler
from .handler import StdDynamicQDQCastONNXActQuantProxyHandler
from .handler import StdQCDQCastONNXActQuantProxyHandler
from .handler import StdQCDQCastONNXQuantLSTMLayerHandler
from .handler import StdQCDQCastONNXTruncQuantProxyHandler
from .handler import StdQCDQCastONNXWeightQuantProxyHandler


class StdQCDQONNXManager(StdONNXBaseManager):
    target_name = 'StdQCDQONNX'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",  # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = {
        StdCDQCastONNXWeightQuantProxyHandler,
        StdCDQCastONNXBiasQuantProxyHandler,
        StdQCDQCastONNXActQuantProxyHandler,
        StdCDQCastONNXDecoupledWeightQuantProxyHandler,
        StdDynamicQDQCastONNXActQuantProxyHandler,
        StdQCDQCastONNXTruncQuantProxyHandler,
        StdCDQCastONNXDecoupledWeightQuantWithInputProxyHandler,
        StdQCDQCastONNXQuantLSTMLayerHandler}

    custom_fns = [
        DebugMarkerFunction,
        QuantizeLinearFn,
        DynamicQuantizeLinearFn,
        DequantizeLinearFn,
        IntClipFn,
        LSTMCellFn,]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)

    @classmethod
    def change_weight_handler(cls, export_quantize_node_weight: bool = False):
        if export_quantize_node_weight:
            cls.handler.discard(StdCDQCastONNXWeightQuantProxyHandler)
            cls.handler.add(StdQCDQCastONNXWeightQuantProxyHandler)
        else:
            cls.handler.discard(StdQCDQCastONNXWeightQuantProxyHandler)
            cls.handler.add(StdCDQCastONNXWeightQuantProxyHandler)
