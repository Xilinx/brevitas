# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Type, Union

from torch import Tensor
from torch.nn import Module

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor

from .quant_layer import ActQuantType
from .quant_layer import QuantInputOutputLayer


class QuantEltwiseAdd(QuantInputOutputLayer, Module):

    def __init__(
            self,
            input_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            output_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            tie_input_output_quant=False,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        Module.__init__(self)
        QuantInputOutputLayer.__init__(
            self, input_quant, output_quant, tie_input_output_quant, return_quant_tensor, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(self, input: Union[Tensor, QuantTensor],
                other: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        input = self.unpack_input(input)
        other = self.unpack_input(other)
        if self.export_mode:
            assert self.cache_quant_io_metadata_only, "Can't cache multiple inputs"
            out = self.export_handler(inp=input.value, other=other.value)
            self._set_global_is_quant_layer(False)
            return out
        quant_input = self.input_quant(input)
        quant_other = self.input_quant(other)
        output = quant_input + quant_other
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)


class QuantCat(QuantInputOutputLayer, Module):

    def __init__(
            self,
            input_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            output_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            tie_input_output_quant: bool = False,
            return_quant_tensor: bool = False,
            **kwargs):
        Module.__init__(self)
        QuantInputOutputLayer.__init__(
            self, input_quant, output_quant, tie_input_output_quant, return_quant_tensor, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(self,
                tensor_list: Union[List[Tensor], List[QuantTensor]],
                dim: int = 1) -> Union[Tensor, QuantTensor]:
        quant_tensor_list = [self.unpack_input(t) for t in tensor_list]
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler([qt.value for qt in quant_tensor_list])
            self._set_global_is_quant_layer(False)
            return out
        quant_tensor_list = [self.input_quant(qt) for qt in quant_tensor_list]
        # trigger an assert if scale factors and bit widths are None or different
        output = QuantTensor.cat(quant_tensor_list, dim=dim)
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)
