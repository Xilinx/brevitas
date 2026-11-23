# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.parameter import ModelWeightParameter

from brevitas.graph.hadamard import fast_hadamard_transform
from brevitas.graph.hadamard import get_hadK
from brevitas.graph.hadamard import matmul_hadU
from brevitas.graph.hadamard import matmul_hadU_cuda

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import GroupwiseIntWeightInferenceHandler
from ..handler import IntInferenceHandler
from ..handler import IntWeightInferencetHandler
from .handler import vLLMDynamicPerRowFloatInferenceHandler
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseFloatWeightInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler
from .handler import vLLMGroupwiseIntWeightInferenceHandler

class_mapping = {
    'vLLMGroupwiseFloatInferenceHandler': vLLMGroupwiseFloatInferenceHandler,
    'vLLMGroupwiseIntInferenceHandler': vLLMGroupwiseIntInferenceHandler,
    'vLLMGroupwiseIntWeightInferenceHandler': vLLMGroupwiseIntWeightInferenceHandler,
    'vLLMGroupwiseFloatWeightInferenceHandler': vLLMGroupwiseFloatWeightInferenceHandler,
    'GroupwiseIntWeightInferenceHandler': GroupwiseIntWeightInferenceHandler,
    'GroupwiseFloatWeightInferenceHandler': GroupwiseFloatWeightInferenceHandler,
    'FloatInferencetHandler': FloatInferencetHandler,
    'FloatWeightInferencetHandler': FloatWeightInferencetHandler,
    'IntWeightInferencetHandler': IntWeightInferencetHandler,
    'IntInferenceHandler': IntInferenceHandler,
    'vLLMDynamicPerRowFloatInferenceHandler': vLLMDynamicPerRowFloatInferenceHandler}


def configure_proxy(quant_config: Optional[Dict[str, Any]]) -> Optional[torch.nn.Module]:
    # No config, no quantizer
    if quant_config is None:
        return None

    # vLLM can query the same layer more than once, so the shared config must not be modified
    quant_config = dict(quant_config)

    # Extract element that are not part of the state dict
    quant_class_name = quant_config.pop('class_type')
    float_to_int_impl_type = quant_config.pop('float_to_int_impl_type')
    scaling_restriction = quant_config.pop('scaling_restriction')
    threshold_restriction = quant_config.pop('threshold_restriction')

    # Scale and zero-point are the only float elements in the state dict
    for k, v in quant_config.items():
        if k == 'scale' or k == 'zero_point':
            quant_config[k] = torch.tensor(v)
        else:
            quant_config[k] = torch.tensor(v, dtype=torch.int)

    # Shapes must be set otherwise the state dict loading will fail
    scale = quant_config.get('scale', None)
    zero_point = quant_config.get('zero_point', None)
    quant_class = class_mapping[quant_class_name]
    if scale is None and zero_point is None:
        quantizer = quant_class()
    else:
        quantizer = quant_class(scale_shape=scale.shape, zero_point_shape=zero_point.shape)

    # Set the remaining attributes
    quantizer.float_to_int_impl_type = float_to_int_impl_type
    if scaling_restriction is not None:
        quantizer.scaling_restriction = scaling_restriction
    if threshold_restriction is not None:
        quantizer.threshold_restriction = threshold_restriction
    quantizer.load_state_dict(quant_config)
    return quantizer


def configure_rotation(
        rotation_config: Optional[Dict[str,
                                       Any]]) -> Optional[Tuple[int, Optional[torch.Tensor], int]]:
    # The Hadamard matrix is rebuilt from the block size. It is computed once, since get_hadK
    # loads the matrices from disk.
    if rotation_config is None:
        return None
    hidden_dim = rotation_config['hidden_dim']
    return (hidden_dim, *get_hadK(hidden_dim))


def rotate(x: torch.Tensor, rotation: Tuple[int, Optional[torch.Tensor], int]) -> torch.Tensor:
    # Block rotations apply the same Hadamard matrix to each block of hidden_dim input channels,
    # which is a no-op reshape for full-vector rotations
    hidden_dim, had_mat, k = rotation
    init_shape = x.shape
    x = x.reshape(*init_shape[:-1], init_shape[-1] // hidden_dim, hidden_dim)
    if fast_hadamard_transform is None:
        x = matmul_hadU(x, hadK=had_mat, K=k)
    else:
        x = matmul_hadU_cuda(x, had_mat, k)
    return x.reshape(init_shape)


def configure_proxies(quant_config):
    # A list has the config of each part of a layer that vLLM merges (e.g., q/k/v projections)
    if isinstance(quant_config, list):
        return [configure_proxy(config) for config in quant_config]
    return configure_proxy(quant_config)


class QuantLinear(LinearMethodBase):

    def __init__(self, quant_configs: Dict[str, Any]) -> None:
        self.input_quant = configure_proxies(quant_configs["input_config"])
        self.weight_quant = configure_proxies(quant_configs["weight_config"])
        self.bias_quant = configure_proxy(quant_configs["bias_config"])
        self.output_quant = configure_proxy(quant_configs["output_config"])
        self.rotation = configure_rotation(quant_configs["rotation_config"])

    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: List[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        self.output_partition_sizes = output_partition_sizes
        for name, quant in (('input', self.input_quant), ('weight', self.weight_quant)):
            if isinstance(quant, list) and len(quant) != len(output_partition_sizes):
                raise ValueError(
                    f"The layer has {len(output_partition_sizes)} parts, but {len(quant)} {name} "
                    "quantizers were exported for them.")

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, module: torch.nn.Module) -> None:
        weight = module.weight.data
        start_idx = 0
        for i, partition_size in enumerate(self.output_partition_sizes):
            end_idx = start_idx + partition_size
            if isinstance(self.weight_quant, list):
                weight_quant = self.weight_quant[i]
            else:
                weight_quant = self.weight_quant
            if weight_quant is not None:
                weight[start_idx:end_idx] = weight_quant(weight[start_idx:end_idx])[0]
            start_idx = end_idx

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotation is not None:
            x = rotate(x, self.rotation)
        if bias is not None and self.bias_quant is not None:
            bias = self.bias_quant(bias)[0]
        if isinstance(self.input_quant, list):
            y = self.apply_per_part(layer, x, bias)
        else:
            if self.input_quant is not None:
                x = self.input_quant(x)[0]
            y = torch.nn.functional.linear(x, layer.weight, bias)
        if self.output_quant is not None:
            y = self.output_quant(y)[0]
        return y

    def apply_per_part(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The parts of a merged layer share the input, but each one has its own input quantizer
        outputs = []
        start_idx = 0
        for input_quant, partition_size in zip(self.input_quant, self.output_partition_sizes):
            end_idx = start_idx + partition_size
            part_x = x if input_quant is None else input_quant(x)[0]
            part_bias = None if bias is None else bias[start_idx:end_idx]
            outputs.append(
                torch.nn.functional.linear(part_x, layer.weight[start_idx:end_idx], part_bias))
            start_idx = end_idx
        return torch.cat(outputs, dim=-1)


class QuantEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Quantization method for the input embedding and the LM head (ParallelLMHead)."""

    def __init__(self, quant_configs: Dict[str, Any]) -> None:
        self.weight_quant = configure_proxy(quant_configs["weight_config"])
        self.input_quant = configure_proxy(quant_configs["input_config"])
        self.rotation = configure_rotation(quant_configs["rotation_config"])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.weight_quant is not None:
            # vLLM pads the vocabulary, while quantization parameters match the original one
            weight = layer.weight.data[:layer.org_vocab_size]
            weight.copy_(self.weight_quant(weight)[0])
        super().process_weights_after_loading(layer)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotation is not None:
            x = rotate(x, self.rotation)
        if self.input_quant is not None:
            x = self.input_quant(x)[0]
        return super().apply(layer, x, bias)


class QuantAttentionMethod(BaseKVCacheMethod):
    """
    Static per-tensor fake quantization of the post-RoPE query, key, and value inputs to attention.
    Keys and values are quantized before being written to the KV cache.
    """

    def __init__(self, quant_config, quant_configs: Dict[str, Any]) -> None:
        super().__init__(quant_config)
        self.q_quant = configure_proxy(quant_configs["q_config"])
        self.k_quant = configure_proxy(quant_configs["k_config"])
        self.v_quant = configure_proxy(quant_configs["v_config"])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.register_forward_pre_hook(self.quantize_qkv)

    def quantize_qkv(self, layer: torch.nn.Module, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        query, key, value, *other_args = args
        if self.q_quant is not None:
            # Brevitas quantizes the query after the softmax scale is applied
            scale = layer.impl.scale
            query = self.q_quant(query * scale)[0] / scale
        if self.k_quant is not None:
            key = self.k_quant(key)[0]
        if self.v_quant is not None:
            value = self.v_quant(value)[0]
        return (query, key, value, *other_args)
