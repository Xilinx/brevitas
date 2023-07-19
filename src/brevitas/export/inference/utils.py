from contextlib import contextmanager

from brevitas.export.manager import _set_proxy_export_mode
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol


@contextmanager
def brevitas_proxy_inference_mode(model, export_manager):
    is_training = model.training
    model.eval()
    model.apply(export_manager.set_export_handler)
    # This should be WeightQuantProxyProtocol but isinstance(BiasQuantProxyProtocol(), WeightQuantProxyProtocol) returns True
    _set_proxy_export_mode(model, True, WeightQuantProxyFromInjector)
    try:
        yield model
    finally:
        _set_proxy_export_mode(model, False, WeightQuantProxyFromInjector)
        model.train(is_training)
