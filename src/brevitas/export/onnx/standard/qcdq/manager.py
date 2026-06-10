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
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
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
    def _validate_dynamo_supported(cls, module: Module):
        # Integer weight/bias export relies on `data_ptr()`, which is incompatible with
        # torch.export (FakeTensor). Surface a clear error for the configurations that
        # would otherwise fail cryptically deep inside the trace.
        for m in module.modules():
            if isinstance(m, DecoupledWeightQuantWithInputProxyFromInjector) and m.is_quant_enabled:
                raise RuntimeError(
                    "QCDQ export with `dynamo=True` does not support input-aware decoupled "
                    "weight quantization (e.g. A2Q): integer weight export uses data_ptr(), "
                    "which is unsupported under torch.export.")
            if isinstance(m, BiasQuantProxyFromInjector) and m.is_quant_enabled:
                raise RuntimeError(
                    "QCDQ export with `dynamo=True` does not support quantized bias: integer "
                    "bias export uses data_ptr(), which is unsupported under torch.export.")

    @classmethod
    def export_onnx(cls, *args, export_weight_q_node: bool = True, **onnx_export_kwargs):
        assert not parse("2.8") > torch_version, f"QCDQ Export with `dynamo=True` only supported for PyTorch>=2.8. Current PyTorch version: {str(torch_version)}"
        assert onnx_export_kwargs["dynamo"]
        # Integer weight export relies on `data_ptr()`, which is incompatible with
        # torch.export (FakeTensor). Require Q-node weight export instead.
        if not export_weight_q_node:
            raise RuntimeError(
                "QCDQ export with `dynamo=True` requires `export_weight_q_node=True`: integer "
                "weight export uses data_ptr(), which is unsupported under torch.export.")
        if args and isinstance(args[0], Module):
            cls._validate_dynamo_supported(args[0])
        key = "optimize"
        wrn_str = f"Optimize=True is recommended with QCDQ export with dynamo=True"
        if key in onnx_export_kwargs.keys():
            if not onnx_export_kwargs[key]:
                logging.warning(wrn_str)
        else:
            logging.warning(wrn_str)
        super(StdQCDQONNXDynamoManager, cls).export_onnx(
            *args, export_weight_q_node=export_weight_q_node, **onnx_export_kwargs)
