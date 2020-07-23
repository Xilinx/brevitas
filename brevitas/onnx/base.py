from typing import Tuple
from abc import ABC, abstractmethod
from copy import deepcopy

try:
    import onnx
    import onnx.optimizer as opt
except ModuleNotFoundError:
    onnx = None
    opt = None

import torch
import torch.onnx

from torch import Tensor
from torch.nn import Module


class BaseHandler(Module, ABC):

    def __init__(self):
        super().__init__()
        self.symbolic_kwargs = {}

    @abstractmethod
    def prepare_for_symbolic_execution(self, module):
        pass

    @abstractmethod
    def symbolic_execution(self, inp: Tensor):
        pass

    def forward(self, inp: Tensor):
        return self.symbolic_execution(inp)


class BaseManager(ABC):

    handlers = []
    model_transforms = []
    onnx_passes = []

    @classmethod
    def handler_from_module(cls, module: Module):
        for handler in cls.handlers:
            if isinstance(module, handler.handled_layer):
                return handler
        raise RuntimeError(f"Module {module.__class__} not supported for export.")

    @classmethod
    def apply_model_transforms(cls, model):
        for tranform in cls.model_transforms:
            model = tranform(model)
        return model

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...],
            export_path: str,
            input_t: Tensor = None,
            torch_onnx_kwargs: dict = None):
        """
        * input_shape : tuple describing the shape of network input e.g. (1, 1, 28, 28)
        * export_path : ONNX filename to export to
        * input_t : if specified, do an initial forward pass with this value. this
                    may be necessary for QuantTensor caching.
        * torch_onnx_kwargs : will be passed as kwargs to torch.onnx.export
        """

        def set_inp_caching_mode(m: Module, enabled: bool):
            if hasattr(m, 'cache_inference_quant_inp'):
                m.cache_inference_quant_inp = enabled

        def set_out_caching_mode(m: Module, enabled: bool):
            if hasattr(m, 'cache_inference_quant_out'):
                m.cache_inference_quant_out = enabled

        def set_export_mode(m: Module, enabled: bool):
            if hasattr(m, 'export_mode'):
                m.export_mode = enabled

        def set_export_handler(m: Module):
            if hasattr(m, 'export_handler') and m.export_handler is None:
                handler = cls.handler_from_module(m)
                m.export_handler = handler()

        if onnx is None or opt is None:
            raise ModuleNotFoundError("Installation of ONNX is required.")
        if torch_onnx_kwargs is None:
            torch_onnx_kwargs = {}

        with torch.no_grad():
            module = module.eval()
            module.apply(set_export_handler)
            if input_t is None:
                input_t = torch.empty(input_shape, dtype=torch.float)
            # do a forward pass with the dummy input to e.g. store input/output shapes
            module.apply(lambda m: set_inp_caching_mode(m, enabled=True))
            module.apply(lambda m: set_out_caching_mode(m, enabled=True))
            _ = module.forward(input_t)
            module.apply(lambda m: set_inp_caching_mode(m, enabled=False))
            module.apply(lambda m: set_out_caching_mode(m, enabled=False))
            # override any given input_t to make sure it's a standard PyTorch tensor
            input_t = torch.empty(input_shape, dtype=torch.float)
            # enable export mode and call export
            module.apply(lambda m: set_export_mode(m, enabled=True))
            torch.onnx.export(module, input_t, export_path, **torch_onnx_kwargs)
            # restore the model to non-export mode to keep it clean
            module.apply(lambda m: set_export_mode(m, enabled=False))
            # do some cleanup on the exported ONNX model
            model = onnx.load(export_path)
            model = opt.optimize(model, cls.onnx_passes)
            model = cls.apply_model_transforms(model)
            onnx.save(model, export_path)