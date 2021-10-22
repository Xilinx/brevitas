from abc import ABC

from torch.nn import functional as F, Module

from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.export.manager import _set_layer_export_handler, _set_layer_export_mode


def _handler_wrapper(handler, cached_io):
    handler = handler()
    handler.prepare_from_cached_io(cached_io)
    return handler


class VitisAIManager(ONNXBaseManager, ABC):

    _fn_to_cache = [
        F.relu,
        F.max_pool2d]

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        _set_layer_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_layer_export_handler(cls, module)

    @classmethod
    def _trace_fn_dispatcher(cls, fn, input, *args, **kwargs):
        handler = cls._fn_cache.pop(0)
        if handler is not None:
            output = handler.cached_symbolic_execution(input, *args, **kwargs)
        else:
            output = fn(input, *args, **kwargs)
        return output