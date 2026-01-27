# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from functools import partial
import json
import os
from typing import Any
from typing import List
from typing import Optional

import torch
from torch.nn import Module
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

import brevitas.config as config
from brevitas.export.inference.vLLM.handler import QuantLinear
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.mixin import QuantLayerMixin
from brevitas.proxy.quant_proxy import QuantProxyFromInjector

from ..manager import _override_act_caching_mode
from ..manager import _override_bias_caching_mode
from ..manager import _override_create_quant_tensor
from ..manager import _override_weight_caching_mode


@register_quantization_config("quant_brevitas")
@dataclass
class QuantConfigBrevitas(QuantizationConfig):

    def __init__(self, ignored_layers: list[str] | None = None, config: str | None = None):
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

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, RowParallelLinear) or isinstance(
                layer, MergedColumnParallelLinear) or isinstance(layer, QKVParallelLinear):
            if self.ignored_layers and is_layer_skipped(
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
                    weight_config = [
                        self.config[layer].get('weight_quant', None) for layer in layers_to_merge]
                    # base_config = combine_configs(self.config, *layers_to_merge)

                return QuantLinear(
                    input_config=input_config,
                    bias_config=bias_config,
                    output_config=output_config,
                    weight_config=weight_config)

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


from json import JSONEncoder

from torch.utils.data import Dataset


class EncodeTensor(JSONEncoder, Dataset):

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.dtype == torch.bfloat16:
                obj = obj.to(torch.float32)
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


class vLLMExportManager():

    wrap_layers = (EqualizedModule, RotatedModule)

    def export(self, model, filepath):
        json_filename = os.path.join(filepath, 'brevitas_config.json')
        config.IGNORE_EXPORT_KEYS = False
        json_to_save = dict()
        proxies_ckpts = os.path.join(filepath, 'brevitas_proxies')
        os.makedirs(proxies_ckpts, exist_ok=True)
        for name, module in model.named_modules():
            if isinstance(module, QuantLayerMixin) or isinstance(module, self.wrap_layers):
                layer_dict = dict()
                json_to_save[name] = layer_dict
                for subname, submodule in module.named_children():
                    if isinstance(submodule, QuantProxyFromInjector) and submodule.is_quant_enabled:
                        proxy_dict = dict()
                        json_to_save[name][subname] = proxy_dict
                        export_handler = submodule.export_handler
                        # torch.save(export_handler.state_dict(), ckpt_path)
                        proxy_dict.update(export_handler.state_dict())
                        proxy_dict['float_to_int_impl_type'] = export_handler.float_to_int_impl_type
                        proxy_dict['class_type'] = export_handler.__class__.__name__
                if isinstance(module, self.wrap_layers):
                    layer_dict['rotation_config'] = dict()
                    layer_dict['rotation_config']['rot_mat_shape'] = module.had_mat.shape[
                        0] if module.had_mat is not None else None
                    layer_dict['rotation_config']['k'] = module.k

        with open(json_filename, 'w') as f:
            json.dump(json_to_save, f, cls=EncodeTensor)
