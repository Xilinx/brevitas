from torch.nn import Module

from brevitas.export.onnx.base import ONNXBaseManager
from brevitas.export.onnx.transform import move_domain_attributes_into_domain
from brevitas.export.base import _set_proxy_export_handler, _set_proxy_export_mode

from .handler import ActQuantProxyHandler, BiasQuantProxyHandler, WeightQuantProxyHandler
from .handler import TruncQuantProxyHandler


class BrevitasONNXManager(ONNXBaseManager):
    target_name = 'brevitas'

    model_transforms = [
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        ActQuantProxyHandler,
        BiasQuantProxyHandler,
        WeightQuantProxyHandler,
        TruncQuantProxyHandler
    ]

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        # proxy level export
        _set_proxy_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        # proxy level export
        _set_proxy_export_handler(cls, module)
