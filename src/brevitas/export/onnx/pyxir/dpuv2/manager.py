from brevitas.export.onnx.base import ONNXBaseManager
from ...transform import move_domain_attributes_into_domain
from .handler import DPUv2QuantConv2dHandler, DPUv2QuantReLUHandler
from .handler import DPUv2QuantEltwiseAddHandler, DPUv2QuantMaxPool2dHandler
from .handler import DPUv2QuantAvgPool2dHandler, DPUv2QuantLinearHandler


class DPUv2Manager(ONNXBaseManager):

    handlers = [
        DPUv2QuantConv2dHandler,
        DPUv2QuantReLUHandler,
        DPUv2QuantEltwiseAddHandler,
        DPUv2QuantAvgPool2dHandler,
        DPUv2QuantLinearHandler,
        DPUv2QuantMaxPool2dHandler]

    model_transforms = [
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]