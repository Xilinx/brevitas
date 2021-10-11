from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from contextlib import ExitStack
from io import BytesIO
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import Module

from brevitas import config
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.jit_utils import jit_patches_generator
from brevitas.nn.mixin.base import _CachedIO, QuantLayerMixin
from brevitas.proxy.quant_proxy import QuantProxyProtocol
from brevitas.utils.python_utils import patch


class _JitTraceExportWrapper(nn.Module):

    def __init__(self, model_to_trace):
        super(_JitTraceExportWrapper, self).__init__()
        self.fn_to_trace = lambda *args, **kwargs: model_to_trace(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.fn_to_trace(*args, **kwargs)


class ExportContext:

    def __init__(self, manager_cls):
        self.target_name = manager_cls.target_name
        self.cache = manager_cls._fn_cache

    def __enter__(self):
        assert config._ONGOING_EXPORT is None
        config._ONGOING_EXPORT = self.target_name
        assert not self.cache

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert config._ONGOING_EXPORT is not None
        config._ONGOING_EXPORT = None
        assert not self.cache


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


def _set_layer_export_mode(m: Module, enabled: bool):
    if isinstance(m, QuantLayerMixin) and hasattr(m, 'export_mode'):
        m.export_mode = enabled


def _set_proxy_export_mode(m: Module, enabled: bool):
    if isinstance(m, QuantProxyProtocol) and hasattr(m, 'export_mode'):
        m.export_mode = enabled


def _set_layer_export_handler(manager_cls, module: Module):
    if (isinstance(module, QuantLayerMixin)
            and hasattr(module, 'export_handler')
            and module.export_handler is None):
        handler = manager_cls.handler_from_module(module)
        if handler is None and module.requires_export_handler:
            raise RuntimeError(f"Module {module.__class__} not supported for export.")
        elif handler is None and not module.requires_export_handler:
            pass
        else:
            module.export_handler = handler()


def _set_proxy_export_handler(manager_cls, module: Module):
    if (isinstance(module, QuantProxyProtocol)
            and hasattr(module, 'export_handler')
            and module.export_handler is None):
        handler = manager_cls.handler_from_module(module, no_inheritance=True)
        module.export_handler = handler()


def _force_requires_grad_false(m: Module):
    backup_dict = {}
    for n, p in m.named_parameters():
        backup_dict[n] = p.requires_grad
        p.requires_grad_(False)
    return backup_dict


def _restore_requires_grad(m: Module, previous_state):
    for n, p in m.named_parameters():
        p.requires_grad_(previous_state[n])


class BaseManager(ABC):

    target_name = None
    handlers = []
    _base_trace_patches_generator = jit_patches_generator
    _fn_to_cache = []
    _fn_cache = []
    _cached_io_handler_map = {}

    @classmethod
    @abstractmethod
    def export(cls, *args, **kwargs):
        return

    @classmethod
    def _gen_patches(cls, fn_dispatcher):
        patches = []
        for fn in cls._fn_to_cache:
            dispatcher = partial(fn_dispatcher, fn)
            p = patch(torch.nn.functional, fn.__name__, dispatcher)
            patches.append(p)
        return patches

    @classmethod
    def _trace_patches(cls):
        patches = []
        if cls._base_trace_patches_generator is not None:
            patches += cls._base_trace_patches_generator()
        patches += cls._gen_patches(cls._trace_fn_dispatcher)
        return patches

    @classmethod
    def _cache_patches(cls):
        return cls._gen_patches(cls._cache_fn_dispatcher)

    @classmethod
    def _restore_fn_patches(cls):
        return [patch(torch.nn.functional, fn.__name__, fn) for fn in cls._fn_to_cache]

    @classmethod
    def _trace_fn_dispatcher(cls, fn, input, *args, **kwargs):
        # baseline impl
        cls._fn_to_cache.pop(0)
        return fn(input, *args, **kwargs)

    @classmethod
    def _cache_fn_dispatcher(cls, fn, input, *args, **kwargs):
        with ExitStack() as stack:
            # disable recursing into this patch
            for mgr in cls._restore_fn_patches():
                stack.enter_context(mgr)
            if isinstance(input, QuantTensor):
                inp_cache = None
                out_cache = None
                if input.is_not_none:
                    inp_cache = _CachedIO(input, metadata_only=True)
                output = fn(input, *args, **kwargs)
                if isinstance(output, QuantTensor) and output.is_not_none:
                    out_cache = _CachedIO(output, metadata_only=True)
                cached_io = (inp_cache, out_cache)
                if fn in cls._cached_io_handler_map:
                    cached_io = cls._cached_io_handler_map[fn](cached_io)
                cls._fn_cache.append(cached_io)
            else:
                # could be a fn invoked within a quant module on a dequant tensor
                # or a function invoked on a float tensor. The former won't show
                # up during jit tracing as they are replaced by symbolic functions,
                # but the latter will, so we have to account for them in the _fn_cache
                output = fn(input, *args, **kwargs)
                if not config._IS_INSIDE_QUANT_LAYER:
                    cls._fn_cache.append(None)
        return output

    @classmethod
    def handler_from_module(cls, module: Module, no_inheritance=False):
        for handler in cls.handlers:
            if no_inheritance:
                if type(module) == handler.handled_layer:
                    return handler
            else:
                if isinstance(module, handler.handled_layer):
                    return handler
        return None

    @classmethod
    @abstractmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        pass

    @classmethod
    @abstractmethod
    def set_export_handler(cls, module: Module):
        pass

    @classmethod
    def _cache_inp_out(cls, module, input_t):
        # force enable caching
        module.apply(lambda m: _override_quant_metadata_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_bias_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_inp_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_out_caching_mode(m, enabled=True))
        with ExitStack() as stack:
            for mgr in cls._cache_patches():
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
            cls._cache_inp_out(module, input_t)
            # unpack quant tensor
            if isinstance(input_t, QuantTensor):
                input_t = input_t.value
            # enable export mode, this triggers collecting export values into handlers
            module.apply(lambda m: cls.set_export_mode(m, enabled=True))
            # force requires_grad to False to let the wrapped model lambda go through tracing
            requires_grad_backup_dict = _force_requires_grad_false(module)
            with ExitStack() as stack:
                for mgr in cls._trace_patches():
                    stack.enter_context(mgr)
                # wrapping with a lambda forces inlining during tracing,
                # converts everything to const and removes unused params/buffers
                traced_model = torch.jit.trace(_JitTraceExportWrapper(module), input_t)
            # Hack to clone the function, otherwise restoring requires_grad
            # on module will break traced_model
            with BytesIO() as tmp:
                torch.jit.save(traced_model, tmp)
                tmp.seek(0)
                traced_model = torch.jit.load(tmp)
            _restore_requires_grad(module, requires_grad_backup_dict)
            module.apply(lambda m: cls.set_export_mode(m, enabled=False))
            module.train(training_state)
            return traced_model