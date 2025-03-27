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
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.logging import setup_logger

from .function import BrevitasBinaryQuantFn
from .function import BrevitasQuantFn
from .function import BrevitasQuantLSTMCellFn
from .function import BrevitasTruncFn
from .function import DOMAIN_STRING as QONNX_DOMAIN_STRING
from .function import DOMAIN_VERSION as QONNX_DOMAIN_VERSION
from .handler import BrevitasActFloatQuantProxyHandler
from .handler import BrevitasActQuantProxyHandler
from .handler import BrevitasBiasQuantProxyHandler
from .handler import BrevitasDecoupledWeightQuantProxyHandler
from .handler import BrevitasDecoupledWeightQuantWithInputProxyHandler
from .handler import BrevitasQuantLSTMLayerHandler
from .handler import BrevitasTruncQuantProxyHandler
from .handler import BrevitasWeightFloatQuantProxyHandler
from .handler import BrevitasWeightQuantProxyHandler

logging = setup_logger(__name__)


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
        BrevitasDecoupledWeightQuantWithInputProxyHandler,
        BrevitasTruncQuantProxyHandler,
        BrevitasQuantLSTMLayerHandler,
        BrevitasWeightFloatQuantProxyHandler,
        BrevitasActFloatQuantProxyHandler]

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

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            args: Optional[Union[Tensor, QuantTensor, Tuple]],
            export_path: Optional[str],
            input_shape: Optional[Tuple[int, ...]],
            input_t: Optional[Union[Tensor, QuantTensor]],
            disable_warnings,
            **onnx_export_kwargs):
        key = "custom_opsets"
        if key in onnx_export_kwargs.keys():
            if QONNX_DOMAIN_STRING in onnx_export_kwargs[key].keys():
                logging.warning(
                    f"Overriding {key}[\"{QONNX_DOMAIN_STRING}\"] = {QONNX_DOMAIN_VERSION}")
            onnx_export_kwargs[key][QONNX_DOMAIN_STRING] = QONNX_DOMAIN_VERSION
        else:
            onnx_export_kwargs[key] = {QONNX_DOMAIN_STRING: QONNX_DOMAIN_VERSION}
        return super(QONNXManager, cls).export_onnx(
            module, args, export_path, input_shape, input_t, disable_warnings, **onnx_export_kwargs)
