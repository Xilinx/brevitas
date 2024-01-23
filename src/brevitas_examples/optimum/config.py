from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BrevitasQuantizationConfig:
    quantization_format: Literal['eager_mode', 'graph_mode'] = 'eager_mode'
    replace_mha_with_quantizable: bool = False
    seqlen: int = 2048
    nsamples: int = 128
    weight_bit_width: int = 8
    input_bit_width: Optional[int] = 8
    weight_param_method: Literal['stats', 'mse'] = 'stats'
    weight_quant_type: Literal['asym', 'sym'] = 'sym'
    scale_precision: Literal['float_scale', 'po2_scale'] = 'float_scale'
    weight_quant_granularity: Literal['per_tensor', 'per_channel', 'per_group'] = 'per_tensor'
    weight_group_size: int = 128
    quantize_zero_point: bool = True
    input_param_method: Literal['stats', 'mse'] = 'stats'
    input_scale_type: Literal['static', 'dynamic'] = 'dynamic'
    input_quant_type: Literal['sym', 'asym'] = 'asym'
    input_quant_granularity: Literal['per_tensor', 'per_row', 'per_group'] = 'per_tensor'
    input_group_size: int = 64
    apply_act_equalization: Literal[None, 'layerwise', 'fx'] = 'fx'
    apply_weight_equalization: bool = False
    apply_gptq: bool = False
