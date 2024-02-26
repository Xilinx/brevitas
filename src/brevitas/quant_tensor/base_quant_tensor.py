from typing import NamedTuple, Optional

from torch import Tensor


class QuantTensorBase(NamedTuple):
    value: Tensor
    scale: Optional[Tensor]
    zero_point: Optional[Tensor]
    bit_width: Optional[Tensor]
    signed_t: Optional[Tensor]
    training_t: Optional[Tensor]


def _unpack_quant_tensor(input_data):
    if isinstance(input_data, QuantTensorBase):
        return input_data.value
    elif isinstance(input_data, tuple):
        return tuple([_unpack_quant_tensor(v) for v in input_data])
    elif isinstance(input_data, list):
        return [_unpack_quant_tensor(v) for v in input_data]
    elif isinstance(input_data, dict):
        return {k: _unpack_quant_tensor(v) for k, v in input_data.items()}
    else:
        return input_data
