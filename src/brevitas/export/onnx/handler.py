from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from .debug import DebugMarkerFunction
from ..base import BaseHandler


class Kernel1dApplHandler(ABC):

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


class Kernel2dApplHandler(ABC):

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

    def attach_debug_info(self, m):
        self.export_debug_name = m.export_debug_name
        self.debug_input = m.cache_inference_quant_inp and not m.cache_quant_io_metadata_only
        self.debug_output = m.cache_inference_quant_out and not m.cache_quant_io_metadata_only

    def forward(self, inp: Tensor, **kwargs):
        debug_fn = lambda x, name:  DebugMarkerFunction.apply(x, self.export_debug_name + name)
        if self.export_debug_name is not None and self.debug_input:
            inp = debug_fn(inp, '.input')
            kwargs = {k: debug_fn(t, '.input' + str(i)) for i, (k, t) in enumerate(kwargs.items())
                      if isinstance(t, Tensor)}
        out = self.symbolic_execution(inp, **kwargs)
        if self.export_debug_name is not None and self.debug_output:
            out = debug_fn(out, '.output')
        return out


class NoOpHandler(ONNXBaseHandler):

    def prepare_for_export(self, module):
        pass

    def symbolic_execution(self, *args, **kwargs):
        pass

    def forward(self, inp: Tensor, **kwargs):
        return inp