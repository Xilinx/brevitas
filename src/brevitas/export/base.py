from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from contextlib import ExitStack
from io import BytesIO

import torch
from torch import Tensor
from torch.nn import Module

from brevitas import config
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.jit_utils import jit_patches_generator


class ExportContext:

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        assert config._ONGOING_EXPORT is None
        config._ONGOING_EXPORT = self.target

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert config._ONGOING_EXPORT is not None
        config._ONGOING_EXPORT = None


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


def _force_requires_grad_false(m: Module):
    backup_dict = {}
    for n, p in m.named_parameters():
        backup_dict[n] = p.requires_grad
        p.requires_grad_(False)
    return backup_dict


def _restore_requires_grad(m: Module, previous_state):
    for n, p in m.named_parameters():
        p.requires_grad_(previous_state[n])


class BaseHandler(Module, ABC):

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        pass

    def reset(self):
        pass


class BaseManager(ABC):

    target_name = None
    handlers = []
    base_trace_patches_generator = jit_patches_generator
    trace_patches_generator = None
    cache_patches_generator = None

    @classmethod
    @abstractmethod
    def export(cls, *args, **kwargs):
        return

    @classmethod
    def trace_patches(cls):
        patches = []
        if cls.base_trace_patches_generator is not None:
            patches += cls.base_trace_patches_generator()
        if cls.trace_patches_generator is not None:
            patches += cls.trace_patches_generator()
        return patches

    @classmethod
    def cache_patches(cls):
        patches = []
        if cls.cache_patches_generator is not None:
            patches += cls.cache_patches_generator()
        return patches

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
    def cache_inp_out(cls, module, input_t):
        # force enable caching
        module.apply(lambda m: _override_quant_metadata_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_bias_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_inp_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_out_caching_mode(m, enabled=True))
        with ExitStack() as stack:
            for mgr in cls.cache_patches():
                stack.enter_context(mgr)
            _ = module.forward(input_t)
        # Restore previous caching properties
        module.apply(lambda m: _restore_quant_metadata_caching_mode(m))
        module.apply(lambda m: _restore_bias_caching_mode(m))
        module.apply(lambda m: _restore_inp_caching_mode(m))
        module.apply(lambda m: _restore_out_caching_mode(m))

    @classmethod
    def jit_inference_trace(cls, module: Module, input_t: Union[Tensor, QuantTensor]):
        with torch.no_grad():
            training_state = module.training
            module = module.eval()
            module.apply(cls.set_export_handler)
            # do a forward pass with the input to e.g. store input/output shapes
            cls.cache_inp_out(module, input_t)
            # unpack quant tensor
            if isinstance(input_t, QuantTensor):
                input_t = input_t.value
            # enable export mode, this triggers collecting export values into handlers
            module.apply(lambda m: _set_export_mode(m, enabled=True))
            # force requires_grad to False to let the wrapped model lambda go through tracing
            requires_grad_backup_dict = _force_requires_grad_false(module)
            with ExitStack() as stack:
                for mgr in cls.trace_patches():
                    stack.enter_context(mgr)
                # wrapping with a lambda forces inlining during tracing,
                # converts everything to const and removes unused params/buffers
                model_fn = lambda *args, **kwargs: module(*args, **kwargs)
                traced_model = torch.jit.trace(model_fn, input_t)
            # Hack to clone the function, otherwise restoring requires_grad
            # on module will break traced_model
            with BytesIO() as tmp:
                torch.jit.save(traced_model, tmp)
                tmp.seek(0)
                traced_model = torch.jit.load(tmp)
            _restore_requires_grad(module, requires_grad_backup_dict)
            module.apply(lambda m: _set_export_mode(m, enabled=False))
            module.train(training_state)
            return traced_model