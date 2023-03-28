# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.manager import ONNXBaseManager

from .function import BrevitasBinaryQuantFn
from .function import BrevitasQuantFn
from .function import BrevitasQuantLSTMCellFn
from .function import BrevitasTruncFn
from .handler import BrevitasActQuantProxyHandler
from .handler import BrevitasBiasQuantProxyHandler
from .handler import BrevitasDecoupledWeightQuantProxyHandler
from .handler import BrevitasQuantLSTMLayerHandler
from .handler import BrevitasTruncQuantProxyHandler
from .handler import BrevitasWeightQuantProxyHandler


class QONNXManager(ONNXBaseManager):
    target_name = 'brevitas'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",  # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        BrevitasActQuantProxyHandler,
        BrevitasBiasQuantProxyHandler,
        BrevitasWeightQuantProxyHandler,
        BrevitasDecoupledWeightQuantProxyHandler,
        BrevitasTruncQuantProxyHandler,
        BrevitasQuantLSTMLayerHandler]

    custom_fns = [
        DebugMarkerFunction,
        BrevitasQuantFn,
        BrevitasBinaryQuantFn,
        BrevitasTruncFn,
        BrevitasQuantLSTMCellFn]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)
