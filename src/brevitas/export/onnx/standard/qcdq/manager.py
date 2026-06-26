# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.nn import Module

from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.function import LSTMCellFn
from brevitas.export.onnx.manager import ONNXDynamoExportMixin
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector

from ..function import DequantizeLinearTorchScriptFn
from ..function import DynamicQuantizeLinearTorchScriptFn
from ..function import IntClipTorchScriptFn
from ..function import QuantizeLinearTorchScriptFn
from ..manager import StdONNXBaseManager
from .handler import StdCDQCastONNXBiasQuantProxyHandler
from .handler import StdDynamicQDQCastONNXActQuantProxyHandler
from .handler import StdFloatQCDQCastONNXActQuantProxyHandler
from .handler import StdFloatQCDQCastONNXWeightQuantProxyHandler
from .handler import StdQCDQCastONNXActQuantProxyHandler
from .handler import StdQCDQCastONNXDecoupledWeightQuantProxyHandler
from .handler import StdQCDQCastONNXDecoupledWeightQuantWithInputProxyHandler
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

    handlers = [
        StdQCDQCastONNXWeightQuantProxyHandler,
        StdFloatQCDQCastONNXWeightQuantProxyHandler,
        StdCDQCastONNXBiasQuantProxyHandler,
        StdQCDQCastONNXActQuantProxyHandler,
        StdFloatQCDQCastONNXActQuantProxyHandler,
        StdQCDQCastONNXDecoupledWeightQuantProxyHandler,
        StdDynamicQDQCastONNXActQuantProxyHandler,
        StdQCDQCastONNXTruncQuantProxyHandler,
        StdQCDQCastONNXDecoupledWeightQuantWithInputProxyHandler,
        StdQCDQCastONNXQuantLSTMLayerHandler]

    custom_fns = [
        DebugMarkerFunction,
        QuantizeLinearTorchScriptFn,
        DynamicQuantizeLinearTorchScriptFn,
        DequantizeLinearTorchScriptFn,
        IntClipTorchScriptFn,
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
    def export_onnx(cls, *args, export_weight_q_node: bool = False, **kwargs):
        cls.change_weight_export(export_weight_q_node)
        super().export_onnx(*args, **kwargs)

    @classmethod
    def change_weight_export(cls, export_weight_q_node: bool = False):
        for handler in cls.handlers:
            if hasattr(handler, '_export_q_node'):
                handler._export_q_node = export_weight_q_node


class StdQCDQONNXDynamoManager(ONNXDynamoExportMixin, StdQCDQONNXManager):

    @classmethod
    def _validate_dynamo_supported(cls, module: Module, export_weight_q_node: bool):
        # Integer weight/bias export relies on `data_ptr()`, which is incompatible with
        # torch.export (FakeTensor).
        for m in module.modules():
            if isinstance(m, DecoupledWeightQuantWithInputProxyFromInjector) and m.is_quant_enabled:
                raise RuntimeError(
                    "QCDQ export with `dynamo=True` does not support input-aware decoupled "
                    "weight quantization (e.g. A2Q).")
            if isinstance(m, BiasQuantProxyFromInjector) and m.is_quant_enabled:
                raise RuntimeError(
                    "QCDQ export with `dynamo=True` does not support quantized bias.")
            if (not export_weight_q_node and isinstance(m, WeightQuantProxyFromInjector) and
                    m.is_quant_enabled):
                raise RuntimeError(
                    "QCDQ export with `dynamo=True` requires `export_weight_q_node=True` for "
                    "quantized weights.")

    @classmethod
    def export_onnx(cls, *args, export_weight_q_node: bool = True, **onnx_export_kwargs):
        cls._check_dynamo_export_kwargs(onnx_export_kwargs)
        if args and isinstance(args[0], Module):
            cls._validate_dynamo_supported(args[0], export_weight_q_node)
        super(StdQCDQONNXDynamoManager, cls).export_onnx(
            *args, export_weight_q_node=export_weight_q_node, **onnx_export_kwargs)
