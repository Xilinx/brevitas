from abc import ABC

from torch.nn import functional as F

from brevitas.export.onnx.base import ONNXBaseManager
from brevitas.export.onnx.transform import move_domain_attributes_into_domain


def _handler_wrapper(handler, cached_io):
    handler = handler()
    handler.prepare_from_cached_io(cached_io)
    return handler


class VitisAIManager(ONNXBaseManager, ABC):

    model_transforms = [
        move_domain_attributes_into_domain]

    _fn_to_cache = [
        F.relu,
        F.max_pool2d]

    @classmethod
    def _trace_fn_dispatcher(cls, fn, input, *args, **kwargs):
        handler = cls._fn_cache.pop(0)
        if handler is not None:
            output = handler.cached_symbolic_execution(input, *args, **kwargs)
        else:
            output = fn(input, *args, **kwargs)
        return output