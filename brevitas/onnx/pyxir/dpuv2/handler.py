from abc import ABC
from typing import Tuple
import math

import torch
from torch import Tensor

from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.nn import QuantConv2d, QuantReLU, QuantEltwiseAdd, QuantMaxPool2d
from brevitas.nn import QuantAdaptiveAvgPool2d
from brevitas.onnx.handler import BaseHandler, Kernel2dApplHandler
from ..handler import DPUQuantConv2dHandler
from ..handler import DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAdaptiveAvgPool2dHandler
from ..handler import DPUQuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler
from ..handler import DPUQuantLinearHandler
from .function import DPUv2QuantConv2dPlaceholderFunction
from .function import DPUv2QuantLinearPlaceholderFunction
from .function import DPUv2QuantReLUPlaceholderFunction
from .function import DPUv2QuantEltwiseAddPlaceholderFunction
from .function import DPUv2QuantMaxPoolPlaceholderFunction
from .function import DPUv2QuantAdaptiveAvgPoolPlaceholderFunction


class DPUv2QuantReLUHandler(DPUQuantReLUHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantReLUPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv2QuantEltwiseAddHandler(DPUQuantEltwiseAddHandler):

    def symbolic_execution(self, inp: Tensor, other: Tensor):
        ret = DPUv2QuantEltwiseAddPlaceholderFunction.apply(
            inp, other, *self.symbolic_kwargs.values())
        return ret


class DPUv2QuantMaxPool2dHandler(DPUQuantMaxPool2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantMaxPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv2QuantAdaptiveAvgPool2dHandler(DPUQuantAdaptiveAvgPool2dHandler):
    handled_layer = QuantAdaptiveAvgPool2d

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantAdaptiveAvgPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv2QuantConv2dHandler(DPUQuantConv2dHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantConv2dPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv2QuantLinearHandler(DPUQuantLinearHandler):

    def symbolic_execution(self, inp: Tensor):
        ret = DPUv2QuantLinearPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret



