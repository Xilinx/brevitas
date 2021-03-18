from abc import ABC


from brevitas.export.onnx.vitis_ai import VitisAIManager
from brevitas.export.onnx.transform import move_domain_attributes_into_domain


def _handler_wrapper(handler, cached_io):
    handler = handler()
    handler.prepare_from_cached_io(cached_io)
    return handler


class PyXIRManager(VitisAIManager, ABC):
    target_name = 'PyXIR'

    model_transforms = [
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]