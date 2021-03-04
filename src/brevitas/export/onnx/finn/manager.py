from brevitas.export.onnx.base import ONNXBaseManager

from ..transform import move_domain_attributes_into_domain
from .transform import move_quant_attributes_into_annotations
from .handler.parameter import FINNQuantConv2dHandler, FINNQuantLinearHandler
from .handler.parameter import FINNQuantConv1dHandler
from .handler.act import FINNQuantHardTanhHandler, FINNQuantReLUHandler, FINNQuantIdentityHandler
from .handler.acc import FINNQuantAvgPool2dHandler


class FINNManager(ONNXBaseManager):
    target_name = 'FINN'

    handlers = [
        FINNQuantLinearHandler,
        FINNQuantConv1dHandler,
        FINNQuantConv2dHandler,
        FINNQuantReLUHandler,
        FINNQuantIdentityHandler,
        FINNQuantHardTanhHandler,
        FINNQuantAvgPool2dHandler]

    model_transforms = [
        move_quant_attributes_into_annotations,
        move_domain_attributes_into_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]
