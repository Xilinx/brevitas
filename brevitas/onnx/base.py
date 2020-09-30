from unittest.mock import patch
from inspect import isclass
from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from packaging import version

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

from brevitas.quant_tensor import QuantTensor
from brevitas.utils.jit_utils import jit_trace_patched


def _override_quant_metadata_caching_mode(m: Module, enabled: bool):
    if hasattr(m, 'cache_quant_io_metadata_only'):
        if not hasattr(m, "cache_quant_io_metadata_only_backup"):
            m.cache_quant_io_metadata_only_backup = m.cache_quant_io_metadata_only
            m.cache_quant_io_metadata_only = enabled


def _override_bias_caching_mode(m: Module, enabled: bool):
    if hasattr(m, 'cache_inference_quant_bias'):
        if not hasattr(m, "cache_inference_quant_bias_backup"):
            m.cache_inference_quant_bias_backup = m.cache_inference_quant_bias
            m.cache_inference_quant_bias = enabled


def _override_inp_caching_mode(m: Module, enabled: bool):
    if hasattr(m, 'cache_inference_quant_inp'):
        if not hasattr(m, "cache_inference_quant_inp_backup"):
            m.cache_inference_quant_inp_backup = m.cache_inference_quant_inp
            m.cache_inference_quant_inp = enabled


def _override_out_caching_mode(m: Module, enabled: bool):
    if hasattr(m, 'cache_inference_quant_out'):
        if not hasattr(m, "cache_inference_quant_out_backup"):
            m.cache_inference_quant_out_backup = m.cache_inference_quant_out
            m.cache_inference_quant_out = enabled


def _restore_quant_metadata_caching_mode(m: Module):
    if hasattr(m, "cache_quant_io_metadata_only_backup"):
        m.cache_quant_io_metadata_only = m.cache_quant_io_metadata_only_backup
        del m.cache_quant_io_metadata_only_backup


def _restore_bias_caching_mode(m: Module):
    if hasattr(m, "cache_inference_quant_bias_backup"):
        m.cache_inference_quant_bias = m.cache_inference_quant_bias_backup
        del m.cache_inference_quant_bias_backup


def _restore_inp_caching_mode(m: Module):
    if hasattr(m, "cache_inference_quant_inp_backup"):
        m.cache_inference_quant_inp = m.cache_inference_quant_inp_backup
        del m.cache_inference_quant_inp_backup


def _restore_out_caching_mode(m: Module):
    if hasattr(m, "cache_inference_quant_out_backup"):
        m.cache_inference_quant_out = m.cache_inference_quant_out_backup
        del m.cache_inference_quant_out_backup


def _set_export_mode(m: Module, enabled: bool):
    if hasattr(m, 'export_mode'):
        m.export_mode = enabled


class BaseManager(ABC):

    handlers = []
    model_transforms = []
    onnx_passes = []

    @classmethod
    def handler_from_module(cls, module: Module):
        for handler in cls.handlers:
            if isinstance(module, handler.handled_layer):
                return handler
        return None

    @classmethod
    def set_export_handler(cls, module: Module):
        if hasattr(module, 'export_handler') and module.export_handler is None:
            handler = cls.handler_from_module(module)
            if handler is None and module.requires_export_handler:
                raise RuntimeError(f"Module {module.__class__} not supported for export.")
            elif handler is None and not module.requires_export_handler:
                pass
            else:
                module.export_handler = handler()

    @classmethod
    def apply_model_transforms(cls, model):
        for tranform in cls.model_transforms:
            model = tranform(model)
        return model

    @classmethod
    def cache_inp_out(cls, module, input_t):
        # force enable caching
        module.apply(lambda m: _override_quant_metadata_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_bias_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_inp_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_out_caching_mode(m, enabled=True))
        _ = module.forward(input_t)
        # Restore previous caching properties
        module.apply(lambda m: _restore_quant_metadata_caching_mode(m))
        module.apply(lambda m: _restore_bias_caching_mode(m))
        module.apply(lambda m: _restore_inp_caching_mode(m))
        module.apply(lambda m: _restore_out_caching_mode(m))

    @classmethod
    def solve_keep_initializers_as_inputs(cls, export_kwargs):
        torch_version = version.parse(torch.__version__)
        # See https://github.com/pytorch/pytorch/commit/7583519b870e33ee3182f330c1bb8663559697b6
        if torch_version >= version.parse('1.3.0'):
            export_kwargs['keep_initializers_as_inputs'] = True

    @classmethod
    def solve_enable_onnx_checker(cls, export_kwargs):
        torch_version = version.parse(torch.__version__)
        if torch_version >= version.parse('1.5.0'):
            export_kwargs['enable_onnx_checker'] = False

    @classmethod
    def jit_trace(cls, module: Module, input_t: Union[Tensor, QuantTensor]):
        with torch.no_grad():
            module = module.eval()
            module.apply(cls.set_export_handler)
            # do a forward pass with the dummy input to e.g. store input/output shapes
            cls.cache_inp_out(module, input_t)
            # override any given input_t to make sure it's a standard PyTorch tensor
            input_shape = input_t.shape if isinstance(input_t, Tensor) else input_t.value.shape
            input_t = torch.empty(input_shape, dtype=torch.float)
            # enable export mode, this triggers collecting export values into handlers
            module.apply(lambda m: _set_export_mode(m, enabled=True))
            traced_model = jit_trace_patched(module, input_t)
            module.apply(lambda m: _set_export_mode(m, enabled=False))
            return traced_model

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...],
            export_path: str,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        """
        * input_shape : tuple describing the shape of network input e.g. (1, 1, 28, 28)
        * export_path : ONNX filename to export to
        * input_t : if specified, do an initial forward pass with this value. this
                    may be necessary for QuantTensor caching.
        * torch_onnx_kwargs : will be passed as kwargs to torch.onnx.export
        """

        if onnx is None or opt is None:
            raise ModuleNotFoundError("Installation of ONNX is required.")

        cls.solve_keep_initializers_as_inputs(kwargs)
        cls.solve_enable_onnx_checker(kwargs)

        with torch.no_grad():
            module = module.eval()
            module.apply(cls.set_export_handler)
            if input_t is None:
                input_t = torch.empty(input_shape, dtype=torch.float)
            # do a forward pass with the dummy input to e.g. store input/output shapes
            cls.cache_inp_out(module, input_t)
            # override any given input_t to make sure it's a standard PyTorch tensor
            input_t = torch.empty(input_shape, dtype=torch.float)
            # enable export mode, this triggers collecting export values into handlers
            module.apply(lambda m: _set_export_mode(m, enabled=True))
            # temporarily disable input caching to avoid collectives empty debug values
            module.apply(lambda m: _override_inp_caching_mode(m, enabled=False))
            # perform export pass
            torch.onnx.export(module, input_t, export_path, **kwargs)
            # restore the model to previous properties
            module.apply(lambda m: _restore_inp_caching_mode(m))
            module.apply(lambda m: _set_export_mode(m, enabled=False))
            # do some cleanup on the exported ONNX model
            model = onnx.load(export_path)
            model = opt.optimize(model, cls.onnx_passes)
            model = cls.apply_model_transforms(model)
            onnx.save(model, export_path)