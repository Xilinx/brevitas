# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Union

from packaging.version import parse
import torch
from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.export.inference.manager import _override_create_quant_tensor
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.graph.calibrate import QuantizationStatusManager
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.logging import setup_logger

from ..manager import _override_act_caching_mode
from .function import bipolar_quant_wrapper
from .function import BrevitasBinaryQuantFn
from .function import BrevitasFloatQuantFn
from .function import BrevitasQuantFn
from .function import BrevitasQuantLSTMCellFn
from .function import BrevitasTruncFn
from .function import DOMAIN_STRING as QONNX_DOMAIN_STRING
from .function import DOMAIN_VERSION as QONNX_DOMAIN_VERSION
from .function import float_quant_wrapper
from .function import int_quant_wrapper
from .function import trunc_quant_wrapper
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
        BrevitasFloatQuantFn,
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
        #key = "dynamo"
        #if key in onnx_export_kwargs.keys():
        #    assert not onnx_export_kwargs["dynamo"]
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


try:
    _dynamo_custom_translation_table = {
        torch.ops.qonnx.int_quant.default: int_quant_wrapper,
        torch.ops.qonnx.bipolar_quant.default: bipolar_quant_wrapper,
        torch.ops.qonnx.float_quant.default: float_quant_wrapper,
        torch.ops.qonnx.trunc_quant.default: trunc_quant_wrapper,}
except:
    # TODO: Remove when PyTorch<2.4 deprecated
    # Workaround for older versions of PyTorch, when the custom_ops API doesn't exist
    _dynamo_custom_translation_table = {}


class QONNXDynamoManager(QONNXManager):
    run_onnx_passes = False  # Skip the optimization step from onnxoptimizer. False required for QONNX + Dynamo
    onnx_passes = ["eliminate_unused_initializer"]
    custom_translation_table = _dynamo_custom_translation_table
    custom_fns = []

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        super(QONNXDynamoManager, cls).set_export_mode(model=model, enabled=enabled)
        # TODO: Move to a DynamoExport Mixin?
        if enabled:
            return_quant_tensor_state = QuantizationStatusManager.disable_return_quant_tensor(
                model
            )
            disable_quant_tensor = partial(_override_create_quant_tensor, state=True)
            model.apply(disable_quant_tensor)
            model._brevitas_return_quant_tensor_state = return_quant_tensor_state #  Store this state in the model, to re-enable later

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
        assert not parse("2.6") > torch_version, f"QONNX Export with `dynamo=True` only supported for PyTorch>=2.4. Current PyTorch version: {str(torch_version)}"
        assert onnx_export_kwargs["dynamo"]
        key = "custom_translation_table"
        if key in onnx_export_kwargs.keys():
            for k, v in cls.custom_translation_table:
                if k in onnx_export_kwargs[key].keys():
                    logging.warning(f"Overriding {key}[{k}] = {v}")
                onnx_export_kwargs[key][k] = v
        else:
            onnx_export_kwargs[key] = cls.custom_translation_table
        key = "optimize"
        wrn_str = f"Optimize=True is recommended with QONNX export with dynamo=True"
        if key in onnx_export_kwargs.keys():
            if not onnx_export_kwargs[key]:
                logging.warning(wrn_str)
        else:
            logging.warning(wrn_str)
        model = super(QONNXDynamoManager, cls).export_onnx(
            module, args, export_path, input_shape, input_t, disable_warnings, **onnx_export_kwargs)
        QuantizationStatusManager.restore_return_quant_tensor(module, module._brevitas_return_quant_tensor_state)
        del module._brevitas_return_quant_tensor_state
        return model
