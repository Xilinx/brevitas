# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export of a quantized model for vLLM.

The export does not need vLLM, which is only required to load the exported model (see manager.py),
so that a model can be quantized and exported in an environment where vLLM is not installed.
"""

import json
from json import JSONEncoder
import os
from typing import Any

import torch
from torch.nn import Module
from torch.utils.data import Dataset

import brevitas.config as config
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.mixin import QuantLayerMixin
from brevitas.nn.quant_embedding import QuantEmbedding
from brevitas.proxy.quant_proxy import QuantProxyFromInjector

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import IntInferenceHandler
from ..handler import IntWeightInferencetHandler
from .handler import vLLMDynamicPerRowFloatInferenceHandler
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseFloatWeightInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler
from .handler import vLLMGroupwiseIntWeightInferenceHandler


class EncodeTensor(JSONEncoder, Dataset):

    def default(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.dtype == torch.bfloat16:
                obj = obj.to(torch.float32)
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


class vLLMExportManager(BaseManager):

    handlers = [
        IntInferenceHandler,
        vLLMDynamicPerRowFloatInferenceHandler,
        FloatInferencetHandler,
        IntWeightInferencetHandler,
        FloatWeightInferencetHandler,
        vLLMGroupwiseIntInferenceHandler,
        vLLMGroupwiseIntWeightInferenceHandler,
        vLLMGroupwiseFloatInferenceHandler,
        vLLMGroupwiseFloatWeightInferenceHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)

    @staticmethod
    def handle_wrap_layer(module: Module):
        class_type = type(module)

        def unwrap(self, destination=None, prefix='', keep_vars=False):
            inner_module_prefix = 'layer'
            output_dict = super(class_type, self).state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)
            layer_keys = [k for k in output_dict.keys() if inner_module_prefix in k]
            wrapper_keys = [k for k in output_dict.keys() if inner_module_prefix not in k]
            # For vLLM Export, we only want to export the inner module's state dict, so we remove the wrapper keys
            # The
            for k in wrapper_keys:
                del output_dict[k]

            for k in layer_keys:
                v = output_dict.pop(k)
                output_dict.update({k.replace('layer.', ''): v})
            return output_dict

        module.orig_state_dict = module.state_dict
        module.state_dict = unwrap

    @staticmethod
    def export(model: Module, tokenizer: Any, filepath: str) -> None:
        layers_to_restore = list()
        json_to_save = dict()
        os.makedirs(filepath, exist_ok=True)
        for name, module in model.named_modules():

            if isinstance(module, (QuantLayerMixin, QuantEmbedding, RotatedModule)):
                layer_dict = dict()
                json_to_save[name] = layer_dict
                if isinstance(module, RotatedModule):
                    layers_to_restore.append(module)
                    vLLMExportManager.handle_wrap_layer(module)
                    layer_dict['rotation_config'] = dict()
                    layer_dict['rotation_config'][
                        'rot_mat_shape'] = module.had_mat.shape[0] if getattr(
                            module, 'had_mat', None) is not None else None
                    layer_dict['rotation_config']['k'] = getattr(module, 'k', None)
                    # Size of the Hadamard blocks, equal to the input size for full-vector rotations
                    layer_dict['rotation_config']['hidden_dim'] = module.hidden_dim

                for subname, submodule in module.named_modules():
                    if isinstance(submodule, QuantProxyFromInjector) and submodule.is_quant_enabled:
                        proxy_dict = dict()
                        proxy_name = subname.split('.')[-1]
                        export_handler = submodule.export_handler
                        if export_handler is None:
                            raise RuntimeError(
                                "Quantization configuration currently not supported for vLLM")
                        # Only the tensors of the handler itself. A handler can also keep the quantizer
                        # that it runs in Brevitas (module_forward), which vLLM does not use: its state,
                        # e.g., the parameters of a learned quantizer, would not load in vLLM.
                        proxy_dict.update({
                            key: value for key,
                            value in export_handler.state_dict().items() if '.' not in key})
                        proxy_dict['threshold_restriction'] = getattr(
                            export_handler, 'threshold_restriction', None)
                        proxy_dict['scaling_restriction'] = getattr(
                            export_handler, 'scaling_restriction', None)
                        proxy_dict['float_to_int_impl_type'] = export_handler.float_to_int_impl_type
                        proxy_dict['class_type'] = export_handler.__class__.__name__
                        json_to_save[name][proxy_name] = proxy_dict

        json_filename = os.path.join(filepath, 'brevitas_config.json')
        with open(json_filename, 'w+') as f:
            json.dump(json_to_save, f, cls=EncodeTensor)

        token = config.IGNORE_PROXY_KEYS.set(True)
        model.save_pretrained(filepath)
        tokenizer.save_pretrained(filepath)
        config.IGNORE_PROXY_KEYS.reset(token)

        for layer in layers_to_restore:
            layer.state_dict = layer.orig_state_dict

        # vLLM rejects a folder without it with a generic error, e.g., if save_pretrained is mocked
        if not os.path.isfile(os.path.join(filepath, 'config.json')):
            raise RuntimeError(
                f"model.save_pretrained did not write config.json in {filepath}, so vLLM cannot "
                "load the export.")
