# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.manager import ExportContext

from .handler import TorchCDQCastBiasQuantProxyHandler
from .handler import TorchQCDQCastActQuantProxyHandler
from .handler import TorchQCDQCastDecoupledWeightQuantProxyHandler
from .handler import TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler
from .handler import TorchQCDQCastTruncQuantProxyHandler
from .handler import TorchQCDQCastWeightQuantProxyHandler


class TorchQCDQManager(BaseManager):
    target_name = 'torch'

    handlers = [
        TorchQCDQCastWeightQuantProxyHandler,
        TorchQCDQCastDecoupledWeightQuantProxyHandler,
        TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler,
        TorchQCDQCastActQuantProxyHandler,
        TorchCDQCastBiasQuantProxyHandler,
        TorchQCDQCastTruncQuantProxyHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)

    @classmethod
    def change_weight_export(cls, export_weight_q_node: bool = False):
        for handler in cls.handlers:
            if hasattr(handler, '_export_q_node'):
                handler._export_weight_q_node = export_weight_q_node

    @classmethod
    def export(
            cls,
            module: Module,
            args,
            export_path: Optional[str] = None,
            export_weight_q_node: bool = False):
        cls.change_weight_export(export_weight_q_node=export_weight_q_node)
        with ExportContext(cls):
            traced_module = cls.jit_inference_trace(module, args, export_path)
        return traced_module
