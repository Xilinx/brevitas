# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
import brevitas.config as config
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops_ste import tensor_clamp_ste


class BitWidthConst(brevitas.jit.ScriptModule):
    """
    ScriptModule that returns a constant bit-width wrapped in a float torch.tensor.

    Args:
        bit_width (int): bit-width value.

    Examples:
        >>> bit_width = BitWidthConst(8)
        >>> bit_width()
        tensor(8.)

    Note:
        The bit-width is not part of the Module's state, meaning that it won't be saved as part of
        a checkpoint.

    Note:
        Maps to bit_width_impl_type == BitWidthImplType.CONST == 'CONST' == 'const' in higher-level APIs.
    """

    def __init__(self, bit_width: int) -> None:
        super(BitWidthConst, self).__init__()
        assert isinstance(bit_width, int)
        self.bit_width = StatelessBuffer(torch.tensor(float(bit_width)))

    @brevitas.jit.script_method
    def forward(self) -> Tensor:
        return self.bit_width()


class BitWidthStatefulConst(brevitas.jit.ScriptModule):
    """
    ScriptModule that returns a constant bit-width wrapped in a float torch.tensor but retains the
    bit-width as part of the module state.

    Args:
        bit_width (int): bit-width value.

    Examples:
        >>> bit_width = BitWidthStatefulConst(8)
        >>> bit_width()
        tensor(8.)

    Note:
        The BitWidthStatefulConst is a counterpart to BitWidthConst with the difference that the
        BitWidthStatefulConst retains the bit-width as part of the Module's state. This means that it
        will be saved as part of a checkpoint.

    Note:
        Maps to bit_width_impl_type == BitWidthImplType.STATEFUL_CONST == 'STATEFUL_CONST' ==
        'stateful_const' in higher-level APIs.
    """

    def __init__(self, bit_width: int) -> None:
        super(BitWidthStatefulConst, self).__init__()
        assert isinstance(bit_width, int)
        self.register_buffer("bit_width", torch.tensor(float(bit_width)))

    @brevitas.jit.script_method
    def forward(self) -> Tensor:
        return self.bit_width

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys: list,
            unexpected_keys,
            error_msgs):
        super(BitWidthStatefulConst, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + "bit_width"
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class MsbClampBitWidth(brevitas.jit.ScriptModule):

    def __init__(
            self,
            bit_width_to_remove_impl: Module,
            min_overall_bit_width: int,
            max_overall_bit_width: int) -> None:
        super(MsbClampBitWidth, self).__init__()

        self.min_overall_bit_width = BitWidthConst(min_overall_bit_width)
        self.max_overall_bit_width = BitWidthConst(max_overall_bit_width)
        self.bit_width_to_remove_impl = bit_width_to_remove_impl

    @brevitas.jit.script_method
    def forward(self, input_bit_width: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl()
        output_bit_width = torch.abs(input_bit_width - bit_width_to_remove)
        output_bit_width = tensor_clamp_ste(
            output_bit_width, self.min_overall_bit_width(), self.max_overall_bit_width())
        return output_bit_width
