from abc import ABC, abstractmethod

from torch import Tensor

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.handler import BaseHandler

__all__ = [
    'Kernel1dApplHandlerMixin',
    'Kernel2dApplHandlerMixin',
    'ONNXBaseHandler'
]


class Kernel1dApplHandlerMixin(ABC):

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 2
        else:
            padding = list(module.padding)
            if len(padding) == 1:
                return padding + padding
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride]
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module):
        if isinstance(module.dilation, int):
            return [module.dilation]
        else:
            dilation = list(module.dilation)
            return dilation

    @staticmethod
    def kernel_shape(module):
        if isinstance(module.kernel_size, int):
            return [module.kernel_size]
        else:
            return list(module.kernel_size)


class Kernel2dApplHandlerMixin(ABC):

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 4
        else:
            padding = list(module.padding) + list(module.padding)
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride] * 2
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module):
        if isinstance(module.dilation, int):
            return [module.dilation] * 2
        else:
            return list(module.dilation)

    @staticmethod
    def kernel_shape(module):
        if isinstance(module.kernel_size, int):
            return [module.kernel_size] * 2
        else:
            return list(module.kernel_size)


class ONNXBaseHandler(BaseHandler, ABC):

    def __init__(self):
        super().__init__()
        self.symbolic_kwargs = None
        self.export_debug_name = None
        self.debug_input = False
        self.debug_output = False

    @abstractmethod
    def prepare_for_export(self, module):
        pass

    @abstractmethod
    def symbolic_execution(self, *args, **kwargs):
        pass

    def reset(self):
        self.symbolic_kwargs = None

    def attach_debug_info(self, m):
        self.export_debug_name = m.export_debug_name
        self.debug_input = m.export_input_debug
        self.debug_output = m.export_output_debug

    def forward(self, inp: Tensor, *args, **kwargs):
        debug_fn = lambda x, name:  DebugMarkerFunction.apply(x, self.export_debug_name + name)
        if self.export_debug_name is not None and self.debug_input:
            inp = debug_fn(inp, '.input')
        out = self.symbolic_execution(inp, *args, **kwargs)
        if self.export_debug_name is not None and self.debug_output:
            if isinstance(out, Tensor):
                out = debug_fn(out, '.output')
            elif isinstance(out, tuple) and isinstance(out[0], Tensor):
                out = list(out)
                out[0] = debug_fn(out[0], '.output')
                out = tuple(out)
        return out