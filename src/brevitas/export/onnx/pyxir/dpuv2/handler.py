from torch import Tensor

from ..handler import DPUQuantConv2dHandler
from ..handler import DPUQuantMaxPool2dHandler
from .function import DPUv2QuantConv2dPlaceholderFunction
from .function import DPUv2QuantMaxPoolPlaceholderFunction


class DPUv2QuantMaxPool2dHandler(DPUQuantMaxPool2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantMaxPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret

    def cached_symbolic_execution(self, inp: Tensor, *args, **kwargs):
        solved_kwargs = self._solve_max_pool2d_kwargs(inp, args, kwargs)
        return DPUv2QuantMaxPoolPlaceholderFunction.apply(
            *solved_kwargs.values(), *self.symbolic_kwargs.values())


class DPUv2QuantConv2dHandler(DPUQuantConv2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantConv2dPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


