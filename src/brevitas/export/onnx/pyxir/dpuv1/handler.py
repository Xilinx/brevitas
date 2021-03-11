from torch import Tensor

from ..handler import DPUQuantConv2dHandler
from ..handler import DPUQuantMaxPool2dHandler
from .function import DPUv1QuantMaxPoolPlaceholderFunction
from .function import DPUv1QuantConv2dPlaceholderFunction


class DPUv1QuantMaxPool2dHandler(DPUQuantMaxPool2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantMaxPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret

    def cached_symbolic_execution(self, inp: Tensor, *args, **kwargs):
        solved_kwargs = self._solve_max_pool2d_kwargs(inp, args, kwargs)
        return DPUv1QuantMaxPoolPlaceholderFunction.apply(
            *solved_kwargs.values(), *self.symbolic_kwargs.values())


class DPUv1QuantConv2dHandler(DPUQuantConv2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv1QuantConv2dPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


