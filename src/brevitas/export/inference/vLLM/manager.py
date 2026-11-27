# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

from brevitas.export.inference.vLLM.layer import QuantAttentionMethod
from brevitas.export.inference.vLLM.layer import QuantEmbeddingMethod
from brevitas.export.inference.vLLM.layer import QuantLinear

# The export does not need vLLM, so it lives in export_manager.py. It is also available from here, for
# backward compatibility.
from .export_manager import EncodeTensor  # noqa: F401
from .export_manager import vLLMExportManager  # noqa: F401


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

    def layer_quant_configs(self, prefix: str) -> Dict[str, Any]:
        base_config = self.config.get(prefix) or {}
        return {
            f"{param}_config": base_config.get(f"{param}_quant", None)
            for param in ["weight", "bias", "output", "input", "rotation"]}

    def get_merged_linear_method(self, prefix: str) -> "QuantizeMethodBase":
        """Quantization method of a layer that vLLM merges from layers that are exported separately.

        This is the case of the q/k/v projections (qkv_proj) and of the gate/up projections
        (gate_up_proj). Each part keeps its weight and input quantizers. Parts that were not
        quantized in Brevitas are not exported, and stay unquantized.
        """
        parent, _, suffix = prefix.rpartition('.')
        names = [
            f'{parent}.{name}' if parent else name
            for name in self.packed_modules_mapping.get(suffix, [])]
        part_configs = [self.config.get(name) or {} for name in names]
        if not any(part_configs):
            return UnquantizedLinearMethod()

        # The other quantizers apply to the whole layer, so they must be the same for every part
        layer_configs = [self.layer_quant_configs(name) for name in names]
        for param in ["bias", "output", "rotation"]:
            configs = [layer_config[f"{param}_config"] for layer_config in layer_configs]
            if any(config != configs[0] for config in configs):
                raise NotImplementedError(
                    f"The layers {names} that vLLM merges into {prefix} have different {param} "
                    "quantizers, which is not supported.")

        quant_configs = layer_configs[0]
        quant_configs["weight_config"] = [config.get('weight_quant') for config in part_configs]
        input_configs = [config.get('input_quant') for config in part_configs]
        # If all the parts share the input quantizer, the layer is computed with a single matmul
        if all(config == input_configs[0] for config in input_configs):
            quant_configs["input_config"] = input_configs[0]
        else:
            quant_configs["input_config"] = input_configs
        return QuantLinear(quant_configs=quant_configs)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.ignored_layers is not None and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if prefix in self.config:
                return QuantLinear(quant_configs=self.layer_quant_configs(prefix))
            return self.get_merged_linear_method(prefix)

        # Covers both the input embedding and the LM head (ParallelLMHead)
        elif isinstance(layer, VocabParallelEmbedding) and prefix in self.config:
            return QuantEmbeddingMethod(quant_configs=self.layer_quant_configs(prefix))

        # Q/K/V quantizers are exported from the QuantScaledDotProductAttention module, which
        # matches the name of vLLM's Attention layer
        elif f"{prefix}.k_transposed_quant" in self.config:
            quant_configs = {
                'q_config': self.config[f"{prefix}.q_scaled_quant"].get('act_quant', None),
                'k_config': self.config[f"{prefix}.k_transposed_quant"].get('act_quant', None),
                'v_config': self.config[f"{prefix}.v_quant"].get('act_quant', None)}
            return QuantAttentionMethod(self, quant_configs=quant_configs)

        return None


# vLLM < 0.10.2 does not add custom quantization methods to platforms with a restricted list (ROCm)
if current_platform.supported_quantization and ("quant_brevitas"
                                                not in current_platform.supported_quantization):
    current_platform.supported_quantization.append("quant_brevitas")


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
