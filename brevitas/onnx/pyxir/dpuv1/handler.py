from torch import Tensor

from brevitas.nn import QuantAdaptiveAvgPool2d
from ..handler import DPUQuantConv2dHandler
from ..handler import DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAdaptiveAvgPool2dHandler
from ..handler import DPUQuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler
from ..handler import DPUQuantLinearHandler
from .function import DPUv1QuantLinearPlaceholderFunction
from .function import DPUv1QuantConv2dPlaceholderFunction
from .function import DPUv1QuantReLUPlaceholderFunction
from .function import DPUv1QuantEltwiseAddPlaceholderFunction
from .function import DPUv1QuantMaxPoolPlaceholderFunction
from .function import DPUv1QuantAdaptiveAvgPoolPlaceholderFunction


class DPUv1QuantReLUHandler(DPUQuantReLUHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantReLUPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantEltwiseAddHandler(DPUQuantEltwiseAddHandler):

    def symbolic_execution(self, inp: Tensor, other: Tensor):
        ret = DPUv1QuantEltwiseAddPlaceholderFunction.apply(
            inp, other, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantMaxPool2dHandler(DPUQuantMaxPool2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantMaxPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantAdaptiveAvgPool2dHandler(DPUQuantAdaptiveAvgPool2dHandler):
    handled_layer = QuantAdaptiveAvgPool2d

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantAdaptiveAvgPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantConv2dHandler(DPUQuantConv2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantConv2dPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantLinearHandler(DPUQuantLinearHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantLinearPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


