# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import torch
from torch.nn import Module
import torch.nn as nn

from brevitas.export.inference.handler import DynamicFloatInferenceHandler
from brevitas.export.inference.handler import DynamicIntInferenceHandler
from brevitas.export.inference.handler import FloatInferencetHandler
from brevitas.export.inference.handler import FloatWeightInferencetHandler
from brevitas.export.inference.handler import GroupwiseFloatInferenceHandler
from brevitas.export.inference.handler import GroupwiseFloatWeightInferenceHandler
from brevitas.export.inference.handler import GroupwiseIntInferenceHandler
from brevitas.export.inference.handler import GroupwiseIntWeightInferenceHandler
from brevitas.export.inference.handler import IntInferencetHandler
from brevitas.export.inference.handler import IntWeightInferencetHandler
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.graph.calibrate import QuantizationStatusManager
from brevitas.proxy.quant_proxy import QuantProxyFromInjector


def _override_caching_mode(m: nn.Module, attr: str, enabled: bool, metadata_only: bool = True):
    cache_var = 'cache_inference_quant_' + attr
    cache_var_metadata_only = cache_var + '_metadata_only'
    if hasattr(m, cache_var):
        setattr(m, cache_var, enabled)
        setattr(m, cache_var_metadata_only, metadata_only)


def _override_bias_caching_mode(m: nn.Module, enabled: bool, metadata_only: bool = True):
    _override_caching_mode(m, 'bias', enabled, metadata_only)


def _override_act_caching_mode(m: nn.Module, enabled: bool, metadata_only: bool = True):
    _override_caching_mode(m, 'act', enabled, metadata_only)


def _override_weight_caching_mode(m: nn.Module, enabled: bool, metadata_only: bool = False):
    _override_caching_mode(m, 'weight', enabled, metadata_only)


def _override_create_quant_tensor(m: nn.Module, state: bool):
    if hasattr(m, 'skip_create_quant_tensor'):
        m.skip_create_quant_tensor = state


class quant_inference_mode:

    def __init__(self, model, cache_quant_weight=False, compile=False, enabled=True):
        self.model = model
        self.enabled = enabled
        self.compile = compile
        self.cache_quant_weight = cache_quant_weight
        self.export_manager = InferenceManager
        self.hook_list = []
        self.return_quant_tensor_state = dict()

    def __enter__(self):
        if self.enabled:
            # Register the hook and store it in the list so that it can be removed by the hook itself when called
            handle = self.model.register_forward_hook(self.hook)
            self.hook_list.append(handle)

            # Enable bias for everything. Optionally, store the fully fake-quantized weights
            self.model.apply(
                lambda m: _override_bias_caching_mode(m, enabled=True, metadata_only=True))
            self.model.apply(lambda m: _override_act_caching_mode(m, enabled=True))
            self.model.apply(
                lambda m: _override_weight_caching_mode(
                    m, enabled=True, metadata_only=not self.cache_quant_weight))

            torch._dynamo.reset()

    def __exit__(self, type, value, traceback):
        # Disable all caching
        # deactivate export mode
        # restore return quant tensor
        InferenceManager.set_export_mode(self.model, enabled=False)
        self.model.apply(
            lambda m: _override_bias_caching_mode(m, enabled=False, metadata_only=False))
        self.model.apply(
            lambda m: _override_act_caching_mode(m, enabled=False, metadata_only=False))
        if self.cache_quant_weight:
            self.model.apply(
                lambda m: _override_weight_caching_mode(m, enabled=False, metadata_only=False))
        QuantizationStatusManager.restore_return_quant_tensor(
            self.model, self.return_quant_tensor_state)
        enable_quant_tensor = partial(_override_create_quant_tensor, state=False)
        self.model.apply(enable_quant_tensor)

    def hook(self, module, inp, out):
        # After one forward pass with caching enabled, we can:
        # - Set the model in export mode
        # - Attach export handlers
        # - Disable return quant tensor since all quant metadata is cached
        assert len(self.hook_list) == 1
        self.hook_list[0].remove()
        self.model.apply(InferenceManager.set_export_handler)
        InferenceManager.set_export_mode(self.model, enabled=True)
        self.return_quant_tensor_state = QuantizationStatusManager.disable_return_quant_tensor(
            self.model)
        disable_quant_tensor = partial(_override_create_quant_tensor, state=True)
        self.model.apply(disable_quant_tensor)
        if self.compile:
            # This is needed to avoid too many recompilations during weight quantization
            torch._dynamo.config.force_parameter_static_shapes = False
            for m in self.model.modules():
                if isinstance(m, QuantProxyFromInjector) and hasattr(
                        m, 'compile_quant') and m.is_quant_enabled:
                    m.compile_quant(compile_export=True)


# Inheritance from BaseManager is not techincally needed
class InferenceManager(BaseManager):
    handlers = [
        IntInferencetHandler,
        DynamicIntInferenceHandler,
        DynamicFloatInferenceHandler,
        FloatInferencetHandler,
        IntWeightInferencetHandler,
        FloatWeightInferencetHandler,
        GroupwiseIntInferenceHandler,
        GroupwiseIntWeightInferenceHandler,
        GroupwiseFloatInferenceHandler,
        GroupwiseFloatWeightInferenceHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)
