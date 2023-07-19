from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import BaseManager
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector

from .handler import InferenceWeightProxyHandler


class InferenceWeightProxyManager(BaseManager):
    handlers = [InferenceWeightProxyHandler]

    @classmethod
    def set_export_handler(cls, module):
        if hasattr(module,
                   'requires_export_handler') and module.requires_export_handler and not isinstance(
                       module, (WeightQuantProxyFromInjector)):
            return
        _set_proxy_export_handler(cls, module)
