from typing import List
from typing import Optional

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from brevitas.graph.hadamard import get_hadK
from brevitas.nn.equalized_layer import RotatedModule

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatInferenceHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import IntInferencetHandler
from ..handler import IntWeightInferencetHandler
from vllm.config import ModelConfig
class_mapping = {
    'GroupwiseFloatInferenceHandler': GroupwiseFloatInferenceHandler,
    'GroupwiseFloatWeightInferenceHandler': GroupwiseFloatWeightInferenceHandler,
    'FloatInferencetHandler': FloatInferencetHandler,
    'FloatWeightInferencetHandler': FloatWeightInferencetHandler,
    'IntWeightInferencetHandler': IntWeightInferencetHandler,
    'IntInferencetHandler': IntInferencetHandler,}


class QuantLinear(LinearMethodBase):

    def __init__(
            self,
            input_config=None,
            weight_config=None,
            bias_config=None,
            output_config=None,
            rotation_config=None):
        self.input_quant = self.configure_proxy(input_config)
        if isinstance(weight_config, list):
            self.weight_quant = dict()
            for i, config in enumerate(weight_config):
                self.weight_quant[i] = self.configure_proxy(config)
        else:
            self.weight_quant = self.configure_proxy(weight_config)
        self.bias_quant = self.configure_proxy(bias_config)
        self.output_quant = self.configure_proxy(output_config)
        self.rotation = self.configure_rotation(rotation_config)

    def configure_rotation(self, rotation_config):
        if rotation_config is None:
            return torch.nn.Identity()
        rot_mat_shape = rotation_config['rot_mat_shape']
        k = rotation_config['k']
        if rot_mat_shape is None:
            had_mat = None
        else:
            had_mat, _ = get_hadK(rot_mat_shape)
        return RotatedModule(self, had_mat, k).rotation_forward

    def configure_proxy(self, quant_config):
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
        scale_shape = quant_config.get('scale', torch.tensor(())).shape
        zero_point_shape = quant_config.get('zero_point', torch.tensor(())).shape
        quant_class_type = class_mapping[quant_class_name]
        quant_class = quant_class_type(scale_shape, zero_point_shape)

        # Set the remaining attributes
        quant_class.float_to_int_impl_type = float_to_int_impl_type
        if scaling_restriction is not None:
            quant_class.scaling_restriction = scaling_restriction
        if threshold_restriction is not None:
            quant_class.threshold_restriction = threshold_restriction
        quant_class.float_to_int_impl_type = float_to_int_impl_type
        quant_class.load_state_dict(quant_config)
        return quant_class

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.parameter import ModelWeightParameter
        breakpoint()
        weight_loader = extra_weight_attrs.get("weight_loader")
        self.input_size_per_partition = input_size_per_partition
        self.output_partition_sizes = output_partition_sizes
        out_per_partition = sum(output_partition_sizes)
        w = torch.empty(
            (out_per_partition, input_size_per_partition),
            device="cuda",
            dtype=params_dtype,
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
             )
        layer.register_parameter("weight", weight)
        # layer.weight = torch.nn.Parameter(w, requires_grad=False)

        # # Handling the packed weights for loading
        # base_loader = extra_weight_attrs.get("weight_loader", None)

        # def packed_weight_loader(param, loaded_weight, loaded_shard_id=None, *args, **kwargs):

        #     if loaded_shard_id is not None:
        #         if isinstance(loaded_shard_id, int):
        #             _loaded_shard_id = loaded_shard_id
        #         else:
        #             if loaded_shard_id == "q":
        #                 _loaded_shard_id = 0
        #             elif loaded_shard_id == "k":
        #                 _loaded_shard_id = 1
        #             elif loaded_shard_id == "v":
        #                 _loaded_shard_id = 2
        #             else:
        #                 raise ValueError(f"Invalid loaded_shard_id: {loaded_shard_id}")

        #         logical_widths = list(output_partition_sizes)
        #         start_idx = sum(logical_widths[:_loaded_shard_id])
        #         end_idx = start_idx + logical_widths[_loaded_shard_id]
        #         weight_quant = self.weight_quant[_loaded_shard_id]
        #     else:
        #         start_idx = 0
        #         end_idx = out_per_partition
        #         weight_quant = self.weight_quant
        #     if not isinstance(weight_quant, torch.nn.Identity):
        #         loaded_weight = weight_quant(loaded_weight.cuda())[0].cpu()

        #     if base_loader is not None:
        #         return base_loader(param[start_idx:end_idx], loaded_weight, *args, **kwargs)
        #     param[start_idx:end_idx].data.copy_(loaded_weight)

        # setattr(layer.weight, "weight_loader", packed_weight_loader)

        # # If this layer has bias, allocate it
        # if getattr(layer, "bias", None) is not None:
        #     b = torch.empty((out_per_partition,), device="cuda", dtype=params_dtype)
        #     layer.bias = torch.nn.Parameter(b, requires_grad=False)
        #     base_bias_loader = extra_weight_attrs.get("bias_loader", None)

        #     def packed_bias_loader(param, loaded_bias, *args, **kwargs):
        #         if isinstance(loaded_bias, (list, tuple)):
        #             loaded_bias = torch.cat(list(loaded_bias), dim=0)
        #         if base_bias_loader is not None:
        #             return base_bias_loader(param, loaded_bias, *args, **kwargs)
        #         param.data.copy_(loaded_bias)

        #     setattr(layer.bias, "bias_loader", packed_bias_loader)

        # # Preserve attrs that vLLM weight loaders may attach
        # for k, v in extra_weight_attrs.items():
        #     if k in ("weight_loader", "bias_loader"):
        #         continue
        #     setattr(layer.weight, k, v)

    def process_weights_after_loading(
        model: torch.nn.Module, model_config: ModelConfig, target_device: torch.device
    ) -> None:
        breakpoint()
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.rotation(x)
        x = self.input_quant(x)
        bias = self.bias_quant(bias) if bias is not None else None
        y = x.matmul(layer.weight.t())
        if bias is not None:
            y = y + bias
        y = self.output_quant(y)
        return y
