# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from packaging.version import parse
from torch.nn import Module

from brevitas import torch_version
from brevitas.export.inference.manager import _override_create_quant_tensor
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.function import LSTMCellFn
from brevitas.graph.calibrate import QuantizationStatusManager
from brevitas.utils.logging import setup_logger

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

logging = setup_logger(__name__)


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


class StdQCDQONNXDynamoManager(StdQCDQONNXManager):
    run_onnx_passes = False  # Skip the optimization step from onnxoptimizer. False required to keep ONNX metadata
    onnx_passes = ["eliminate_unused_initializer"]
    custom_fns = []

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        super(StdQCDQONNXDynamoManager, cls).set_export_mode(model=model, enabled=enabled)
        # torch.export cannot trace QuantTensor objects, so we disable their creation
        # during export and restore the original behaviour afterwards.
        if enabled:
            return_quant_tensor_state = QuantizationStatusManager.disable_return_quant_tensor(model)
            disable_quant_tensor = partial(_override_create_quant_tensor, state=True)
            model.apply(disable_quant_tensor)
            model._brevitas_return_quant_tensor_state = return_quant_tensor_state
        else:
            enable_quant_tensor = partial(_override_create_quant_tensor, state=False)
            model.apply(enable_quant_tensor)
            QuantizationStatusManager.restore_return_quant_tensor(
                model, model._brevitas_return_quant_tensor_state)
            del model._brevitas_return_quant_tensor_state

    @classmethod
    def export_onnx(cls, *args, export_weight_q_node: bool = False, **onnx_export_kwargs):
        assert not parse("2.8") > torch_version, f"QCDQ Export with `dynamo=True` only supported for PyTorch>=2.8. Current PyTorch version: {str(torch_version)}"
        assert onnx_export_kwargs["dynamo"]
        key = "optimize"
        wrn_str = f"Optimize=True is recommended with QCDQ export with dynamo=True"
        if key in onnx_export_kwargs.keys():
            if not onnx_export_kwargs[key]:
                logging.warning(wrn_str)
        else:
            logging.warning(wrn_str)
        # Integer weight/bias export relies on `data_ptr()`, which is incompatible with
        # torch.export (FakeTensor). Force Q-node weight export so weights are emitted as
        # QuantizeLinear+DequantizeLinear from their floating-point values instead.
        super(StdQCDQONNXDynamoManager, cls).export_onnx(
            *args, export_weight_q_node=True, **onnx_export_kwargs)
