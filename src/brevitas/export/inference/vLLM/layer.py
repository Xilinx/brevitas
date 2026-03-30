# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

from brevitas.graph.hadamard import get_hadK
from brevitas.nn.equalized_layer import RotatedModule

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import GroupwiseIntWeightInferenceHandler
from ..handler import IntInferenceHandler
from ..handler import IntWeightInferencetHandler
from .handler import vLLMDynamicPerRowFloatInferenceHandler
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler

class_mapping = {
    'vLLMGroupwiseFloatInferenceHandler': vLLMGroupwiseFloatInferenceHandler,
    'vLLMGroupwiseIntInferenceHandler': vLLMGroupwiseIntInferenceHandler,
    'GroupwiseIntWeightInferenceHandler': GroupwiseIntWeightInferenceHandler,
    'GroupwiseFloatWeightInferenceHandler': GroupwiseFloatWeightInferenceHandler,
    'FloatInferencetHandler': FloatInferencetHandler,
    'FloatWeightInferencetHandler': FloatWeightInferencetHandler,
    'IntWeightInferencetHandler': IntWeightInferencetHandler,
    'IntInferenceHandler': IntInferenceHandler,
    'vLLMDynamicPerRowFloatInferenceHandler': vLLMDynamicPerRowFloatInferenceHandler}


class QuantLinear(LinearMethodBase):

    def __init__(self, quant_configs: Optional[Dict[str, Any]] = None) -> None:

        self.input_quant = self.configure_proxy(quant_configs["input_config"])
        weight_config = quant_configs["weight_config"]
        if isinstance(weight_config, list):
            self.weight_quant = {
                i: self.configure_proxy(config) for i, config in enumerate(weight_config)}
        else:
            self.weight_quant = self.configure_proxy(weight_config)
        self.bias_quant = self.configure_proxy(quant_configs["bias_config"])
        self.output_quant = self.configure_proxy(quant_configs["output_config"])
        self.rotation = self.configure_rotation(quant_configs["rotation_config"])

    def configure_rotation(self, rotation_config: Optional[Dict[str,
                                                                Any]]) -> Optional[RotatedModule]:
        if rotation_config is None:
            return None
        rot_mat_shape = rotation_config['rot_mat_shape']
        k = rotation_config['k']
        if rot_mat_shape is None:
            had_mat = None
        else:
            had_mat, _ = get_hadK(rot_mat_shape)
        return RotatedModule(self, had_mat, k)

    def configure_proxy(self, quant_config: Optional[Dict[str, Any]]) -> torch.nn.Module:
        # No config, no quantizer
        if quant_config is None:
            return torch.nn.Identity()

        # Extract element that are not part of the state dict
        quant_class_name = quant_config['class_type']
        float_to_int_impl_type = quant_config['float_to_int_impl_type']
        scaling_restriction = quant_config['scaling_restriction']
        threshold_restriction = quant_config['threshold_restriction']
        del quant_config['class_type']
        del quant_config['float_to_int_impl_type']
        del quant_config['scaling_restriction']
        del quant_config['threshold_restriction']

        # Scale and zero-point are the only float elements in the state dict
        for k, v in quant_config.items():
            if not isinstance(v, torch.Tensor):
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
            scale_shape = scale.shape
            zero_point_shape = zero_point.shape
            quantizer = quant_class(scale_shape=scale_shape, zero_point_shape=zero_point_shape)

        # Set the remaining attributes
        quantizer.float_to_int_impl_type = float_to_int_impl_type
        if scaling_restriction is not None:
            quantizer.scaling_restriction = scaling_restriction
        if threshold_restriction is not None:
            quantizer.threshold_restriction = threshold_restriction
        quantizer.float_to_int_impl_type = float_to_int_impl_type
        quantizer.load_state_dict(quant_config)
        return quantizer

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
        self.input_size_per_partition = input_size_per_partition
        self.output_partition_sizes = output_partition_sizes
        out_per_partition = sum(output_partition_sizes)

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
        for i in range(len(self.output_partition_sizes)):
            logical_widths = list(self.output_partition_sizes)
            start_idx = sum(logical_widths[:i])
            end_idx = start_idx + logical_widths[i]
            if isinstance(self.weight_quant, dict):
                weight_quant = self.weight_quant[i]
            else:
                weight_quant = self.weight_quant

            weight[start_idx:end_idx] = weight_quant(weight[start_idx:end_idx])[0]

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotation is not None:
            x = self.rotation.rotation_forward(x)
        x = self.input_quant(x)[0]
        bias = self.bias_quant(bias) if bias is not None else None
        y = torch.nn.functional.linear(x, layer.weight, bias)
        y = self.output_quant(y)
        return y
