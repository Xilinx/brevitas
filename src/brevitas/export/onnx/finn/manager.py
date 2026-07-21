# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.nn import Module

from brevitas.export.onnx.qonnx.manager import QONNXDynamoManager
from brevitas.export.onnx.qonnx.manager import QONNXManager
from brevitas.nn.target.finn import PWPolyFActivation
from brevitas.utils.logging import setup_logger

from .function import DOMAIN_STRING as FINN_DOMAIN_STRING
from .function import DOMAIN_VERSION as FINN_DOMAIN_VERSION
from .function import FINNPWPolyFTorchScriptFn
from .handler import FINNPWPolyFHandler

__all__ = [
    "FINNONNXDynamoManager",
    "FINNONNXManager",]

logging = setup_logger(__name__)


def _set_pwpolyf_export_mode(model: Module, enabled: bool):
    for module in model.modules():
        if isinstance(module, PWPolyFActivation):
            module.export_mode = enabled


def _set_pwpolyf_export_handler(manager_cls, module: Module):
    if isinstance(module, PWPolyFActivation) and module.export_handler is None:
        handler = manager_cls.handler_from_module(module)
        if handler is None:
            return
        module.export_handler = handler()


def _set_finn_custom_opset(onnx_export_kwargs):
    key = "custom_opsets"
    if onnx_export_kwargs.get(key) is None:
        onnx_export_kwargs[key] = {}
    if FINN_DOMAIN_STRING in onnx_export_kwargs[key]:
        logging.warning(
            f"Overriding {key}[\"{FINN_DOMAIN_STRING}\"] = {FINN_DOMAIN_VERSION}")
    onnx_export_kwargs[key][FINN_DOMAIN_STRING] = FINN_DOMAIN_VERSION


class FINNONNXManager(QONNXManager):
    handlers = QONNXManager.handlers + [FINNPWPolyFHandler]
    custom_fns = QONNXManager.custom_fns + [FINNPWPolyFTorchScriptFn]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        super(FINNONNXManager, cls).set_export_mode(model, enabled)
        _set_pwpolyf_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        super(FINNONNXManager, cls).set_export_handler(module)
        _set_pwpolyf_export_handler(cls, module)

    @classmethod
    def export_onnx(cls, *args, **onnx_export_kwargs):
        _set_finn_custom_opset(onnx_export_kwargs)
        return super(FINNONNXManager, cls).export_onnx(*args, **onnx_export_kwargs)


class FINNONNXDynamoManager(QONNXDynamoManager):
    handlers = QONNXDynamoManager.handlers + [FINNPWPolyFHandler]
    custom_fns = []

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        super(FINNONNXDynamoManager, cls).set_export_mode(model, enabled)
        _set_pwpolyf_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        super(FINNONNXDynamoManager, cls).set_export_handler(module)
        _set_pwpolyf_export_handler(cls, module)

    @classmethod
    def export_onnx(cls, *args, **onnx_export_kwargs):
        _set_finn_custom_opset(onnx_export_kwargs)
        return super(FINNONNXDynamoManager, cls).export_onnx(*args, **onnx_export_kwargs)
