# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from io import BytesIO
from typing import Tuple, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from typing_extensions import Protocol

from brevitas import config
from brevitas.nn.mixin.base import QuantLayerMixin
from brevitas.nn.mixin.base import QuantRecurrentLayerMixin
from brevitas.proxy.quant_proxy import QuantProxyProtocol
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.jit_utils import clear_class_registry
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


def _override_act_caching_mode(m: Module, enabled: bool):
    if hasattr(m, 'cache_inference_quant_act'):
        if not hasattr(m, "cache_inference_quant_act_backup"):
            m.cache_inference_quant_act_backup = m.cache_inference_quant_act
            m.cache_inference_quant_act = enabled


def _restore_quant_metadata_caching_mode(m: Module):
    if hasattr(m, "cache_quant_io_metadata_only_backup"):
        m.cache_quant_io_metadata_only = m.cache_quant_io_metadata_only_backup
        del m.cache_quant_io_metadata_only_backup


def _restore_bias_caching_mode(m: Module):
    if hasattr(m, "cache_inference_quant_bias_backup"):
        m.cache_inference_quant_bias = m.cache_inference_quant_bias_backup
        del m.cache_inference_quant_bias_backup


def _restore_act_caching_mode(m: Module):
    if hasattr(m, "cache_inference_quant_act_backup"):
        m.cache_inference_quant_act = m.cache_inference_quant_act_backup
        del m.cache_inference_quant_act_backup


def _set_recurrent_layer_export_mode(model: Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, QuantRecurrentLayerMixin) and hasattr(m, 'export_mode'):
            m.export_mode = enabled


def _set_layer_export_mode(model: Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, QuantLayerMixin) and hasattr(m, 'export_mode'):
            m.export_mode = enabled


def _set_proxy_export_mode(model: Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, QuantProxyProtocol) and hasattr(m, 'export_mode'):
            m.export_mode = enabled


def _set_export_handler(manager_cls, module: Module, instance_type, no_inheritance):
    if (isinstance(module, instance_type) and hasattr(module, 'export_handler') and
            module.export_handler is None):
        handler = manager_cls.handler_from_module(module, no_inheritance)
        if handler is None and module.requires_export_handler:
            raise RuntimeError(f"Module {module.__class__} not supported for export.")
        elif handler is None and not module.requires_export_handler:
            pass
        else:
            module.export_handler = handler()


def _set_layer_export_handler(manager_cls, module: Module):
    _set_export_handler(manager_cls, module, QuantLayerMixin, no_inheritance=False)


def _set_proxy_export_handler(manager_cls, module: Module):
    _set_export_handler(manager_cls, module, QuantProxyProtocol, no_inheritance=True)


def _set_recurrent_layer_export_handler(manager_cls, module: Module):
    _set_export_handler(manager_cls, module, QuantRecurrentLayerMixin, no_inheritance=True)


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
    _fn_to_cache = []
    _fn_cache = []
    _cached_io_handler_map = {}

    @classmethod
    @abstractmethod
    def export(cls, *args, **kwargs):
        return

    @classmethod
    def _trace_fn_dispatcher(cls, fn, input, *args, **kwargs):
        # baseline impl
        cls._fn_to_cache.pop(0)
        return fn(input, *args, **kwargs)

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
    def set_export_mode(cls, model: Module, enabled: bool):
        pass

    @classmethod
    @abstractmethod
    def set_export_handler(cls, module: Module):
        pass

    @classmethod
    def _cache_inp_out(cls, module, *args, **kwargs):
        # force enable caching
        module.apply(lambda m: _override_quant_metadata_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_bias_caching_mode(m, enabled=True))
        module.apply(lambda m: _override_act_caching_mode(m, enabled=True))
        _ = module.forward(*args, **kwargs)
        # Restore previous caching properties
        module.apply(lambda m: _restore_quant_metadata_caching_mode(m))
        module.apply(lambda m: _restore_bias_caching_mode(m))
        module.apply(lambda m: _restore_act_caching_mode(m))

    @classmethod
    def jit_inference_trace(
            cls, module: Module, args: Union[Tensor, QuantTensor, Tuple], export_path: str = None):
        with torch.no_grad():
            training_state = module.training
            module = module.eval()
            module.apply(cls.set_export_handler)
            # do a forward pass with the input to e.g. store input/output shapes
            if isinstance(args, (Tensor, QuantTensor)):
                cls._cache_inp_out(module, args)
            else:
                cls._cache_inp_out(module, *args)
            # enable export mode, this triggers collecting export values into handlers
            cls.set_export_mode(module, enabled=True)
            # force requires_grad to False to let the wrapped model lambda go through tracing
            requires_grad_backup_dict = _force_requires_grad_false(module)
            # wrapping with a lambda forces inlining during tracing,
            # converts everything to const and removes unused params/buffers
            traced_model = torch.jit.trace(_JitTraceExportWrapper(module), args)
            # Hack to clone the function, otherwise restoring requires_grad
            # on module will break traced_model
            with BytesIO() as tmp:
                torch.jit.save(traced_model, tmp)
                tmp.seek(0)
                traced_model = torch.jit.load(tmp)
                del tmp
            if export_path is not None:
                traced_model.save(export_path)
            # helps fight memory leaks caused by torch.jit.trace
            clear_class_registry()
            _restore_requires_grad(module, requires_grad_backup_dict)
            cls.set_export_mode(module, enabled=False)
            module.train(training_state)
            return traced_model
