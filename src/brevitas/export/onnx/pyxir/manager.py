from abc import ABC

from brevitas.export.onnx.base import ONNXBaseManager

from ..transform import move_domain_attributes_into_domain


class PyXIRManager(ONNXBaseManager, ABC):
    target_name = 'PyXIR'

    model_transforms = [
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]