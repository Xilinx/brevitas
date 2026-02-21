# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from functools import partial
import json
from json import JSONEncoder
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import Dataset
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

import brevitas.config as config
from brevitas.export.inference.vLLM.layer import QuantLinear
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_recurrent_layer_export_handler
from brevitas.export.manager import _set_recurrent_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.mixin import QuantLayerMixin
from brevitas.proxy.quant_proxy import QuantProxyFromInjector

from ..handler import DynamicFloatInferenceHandler
from ..handler import DynamicIntInferenceHandler
from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatInferenceHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import GroupwiseIntInferenceHandler
from ..handler import GroupwiseIntWeightInferenceHandler
from ..handler import IntInferencetHandler
from ..handler import IntWeightInferencetHandler
from ..manager import _override_act_caching_mode
from ..manager import _override_bias_caching_mode
from ..manager import _override_create_quant_tensor
from ..manager import _override_weight_caching_mode
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler


@register_quantization_config("quant_brevitas")
@dataclass
class QuantConfigBrevitas(QuantizationConfig):

    def __init__(self, ignored_layers: list[str] | None = None, config: Dict | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers
        self.config = config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantConfigTcast":
        return cls(config=config)

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability needed for the kernel.
        return 0

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "quant_brevitas"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["brevitas_config.json"]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "QuantizeMethodBase" | None:
        if isinstance(layer, RowParallelLinear) or isinstance(
                layer, MergedColumnParallelLinear) or isinstance(layer, QKVParallelLinear):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            else:

                if prefix in self.config:
                    base_config = self.config[prefix]
                    input_config = base_config.get('input_quant', None)
                    bias_config = base_config.get('bias_quant', None)
                    output_config = base_config.get('output_quant', None)
                    weight_config = base_config.get('weight_quant', None)
                    rotation_config = base_config.get('rotation_config', None)
                else:
                    base = prefix.split('.')[:-1]
                    base = '.'.join(base)
                    suffix = prefix.split('.')[-1]
                    layers_to_merge = self.packed_modules_mapping[suffix]
                    layers_to_merge = [base + '.' + x for x in layers_to_merge]

                    base_config = self.config[layers_to_merge[0]]
                    input_config = base_config.get('input_quant', None)
                    bias_config = base_config.get('bias_quant', None)
                    output_config = base_config.get('output_quant', None)
                    rotation_config = base_config.get('rotation_config', None)
                    weight_config = [
                        self.config[layer].get('weight_quant', None) for layer in layers_to_merge]
                    # base_config = combine_configs(self.config, *layers_to_merge)

                return QuantLinear(
                    input_config=input_config,
                    bias_config=bias_config,
                    output_config=output_config,
                    weight_config=weight_config,
                    rotation_config=rotation_config)

        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()

        return None


def combine_configs(config, *names):
    base_config = config[names[0]]
    scale = None  #base_config['scale']
    for n in names:
        if scale is None:
            scale = torch.tensor(config[n]['weight_quant']['scale'])
        else:
            v = torch.tensor(config[n]['weight_quant']['scale'])
            scale = torch.cat((scale, v), 0)
    base_config['weight_quant']['scale'] = scale
    return base_config


class EncodeTensor(JSONEncoder, Dataset):

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.dtype == torch.bfloat16:
                obj = obj.to(torch.float32)
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


class vLLMExportManager(BaseManager):

    handlers = [
        IntInferencetHandler,
        DynamicIntInferenceHandler,
        DynamicFloatInferenceHandler,
        FloatInferencetHandler,
        IntWeightInferencetHandler,
        FloatWeightInferencetHandler,
        vLLMGroupwiseIntInferenceHandler,
        GroupwiseIntWeightInferenceHandler,
        vLLMGroupwiseFloatInferenceHandler,
        GroupwiseFloatWeightInferenceHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)
        _set_recurrent_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)
        _set_recurrent_layer_export_handler(cls, module)

    wrap_layers = (EqualizedModule, RotatedModule)

    def handle_wrap_layer(self, module: Module):

        def unwrap(self, destination=None, prefix='', keep_vars=False):
            inner_module_prefix = 'layer'
            output_dict = super(RotatedModule, self).state_dict(
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

    def export(self, model, tokenizer, filepath):
        json_filename = os.path.join(filepath, 'brevitas_config.json')
        layers_to_restore = list()
        json_to_save = dict()
        for name, module in model.named_modules():

            if isinstance(module, QuantLayerMixin) or isinstance(module, self.wrap_layers):
                self.handle_wrap_layer(module)
                layers_to_restore.append(module)
                layer_dict = dict()
                json_to_save[name] = layer_dict
                if isinstance(module, self.wrap_layers):
                    layer_dict['rotation_config'] = dict()
                    layer_dict['rotation_config'][
                        'rot_mat_shape'] = module.had_mat.shape[0] if getattr(
                            module, 'had_mat', None) is not None else None
                    layer_dict['rotation_config']['k'] = getattr(module, 'k', None)

                for subname, submodule in module.named_modules():
                    if isinstance(submodule, QuantProxyFromInjector) and submodule.is_quant_enabled:
                        proxy_dict = dict()
                        proxy_name = subname.split('.')[-1]
                        export_handler = submodule.export_handler
                        proxy_dict.update(export_handler.state_dict())
                        proxy_dict['threshold_restriction'] = getattr(
                            export_handler, 'threshold_restriction', None)
                        proxy_dict['scaling_restriction'] = getattr(
                            export_handler, 'scaling_restriction', None)
                        proxy_dict['float_to_int_impl_type'] = export_handler.float_to_int_impl_type
                        proxy_dict['class_type'] = export_handler.__class__.__name__
                        json_to_save[name][proxy_name] = proxy_dict

        with open(json_filename, 'w') as f:
            json.dump(json_to_save, f, cls=EncodeTensor)

        config.IGNORE_EXPORT_KEYS = True
        model.save_pretrained(filepath)
        tokenizer.save_pretrained(filepath)
        config.IGNORE_EXPORT_KEYS = False
        for layer in layers_to_restore:
            layer.state_dict = layer.orig_state_dict
