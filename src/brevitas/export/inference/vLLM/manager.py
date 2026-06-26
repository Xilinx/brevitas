# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.mixin import QuantLayerMixin
from brevitas.proxy.quant_proxy import QuantProxyFromInjector

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import GroupwiseIntWeightInferenceHandler
from ..handler import IntInferenceHandler
from ..handler import IntWeightInferencetHandler
from .handler import vLLMDynamicPerRowFloatInferenceHandler
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler


@register_quantization_config("quant_brevitas")
@dataclass
class QuantConfigBrevitas(QuantizationConfig):

    def __init__(
            self,
            ignored_layers: Optional[List[str]] = None,
            config: Optional[Dict] = None) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantConfigBrevitas":
        return cls(config=config)

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability needed for the kernel.
        return 0

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "quant_brevitas"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["brevitas_config.json"]

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, RowParallelLinear) or isinstance(
                layer, MergedColumnParallelLinear) or isinstance(layer, QKVParallelLinear):
            if self.ignored_layers is not None and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            else:
                if prefix in self.config:
                    base_config = self.config[prefix]
                    quant_configs = {
                        f"{param}_config": base_config.get(f"{param}_quant", None)
                        for param in ["weight", "bias", "output", "input", "rotation"]}

                else:
                    base = prefix.split('.')[:-1]
                    base = '.'.join(base)
                    suffix = prefix.split('.')[-1]
                    layers_to_merge = self.packed_modules_mapping[suffix]
                    layers_to_merge = [base + '.' + x for x in layers_to_merge]

                    base_config = self.config[layers_to_merge[0]]
                    quant_configs = {
                        f"{param}_config": base_config.get(f"{param}_quant", None)
                        for param in ["bias", "output", "input", "rotation"]}
                    weight_config = [
                        self.config[layer].get('weight_quant', None) for layer in layers_to_merge]
                    quant_configs["weight_config"] = weight_config

                return QuantLinear(quant_configs=quant_configs)

        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()

        return None


def combine_configs(config: Dict, *names: str) -> Dict:
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

            if isinstance(module, QuantLayerMixin) or isinstance(module, RotatedModule):
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

                for subname, submodule in module.named_modules():
                    if isinstance(submodule, QuantProxyFromInjector) and submodule.is_quant_enabled:
                        proxy_dict = dict()
                        proxy_name = subname.split('.')[-1]
                        export_handler = submodule.export_handler
                        if export_handler is None:
                            raise RuntimeError(
                                "Quantization configuration currently not supported for vLLM")
                        proxy_dict.update(export_handler.state_dict())
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
