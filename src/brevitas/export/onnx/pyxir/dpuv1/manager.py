from brevitas.export.onnx.base import ONNXBaseManager
from ...transform import move_domain_attributes_into_domain
from .handler import DPUv1QuantConv2dHandler, DPUv1QuantReLUHandler
from .handler import DPUv1QuantEltwiseAddHandler, DPUv1QuantMaxPool2dHandler
from .handler import DPUv1QuantAvgPool2dHandler, DPUv1QuantLinearHandler


class DPUv1Manager(ONNXBaseManager):

    handlers = [
        DPUv1QuantConv2dHandler,
        DPUv1QuantReLUHandler,
        DPUv1QuantEltwiseAddHandler,
        DPUv1QuantAvgPool2dHandler,
        DPUv1QuantLinearHandler,
        DPUv1QuantMaxPool2dHandler]

    model_transforms = [
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]