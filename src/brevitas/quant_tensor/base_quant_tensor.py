from typing import NamedTuple

from torch import Tensor


class QuantTensorBase(NamedTuple):
    value: Tensor
    scale: Tensor
    zero_point: Tensor
    bit_width: Tensor
    signed_t: Tensor
    training_t: Tensor


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
