# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Union, Type, List, Optional

from torch import Tensor
from torch.nn import Module

from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8ActPerTensorFloat
from .quant_layer import QuantInputOutputLayer, ActQuantType


class QuantEltwiseAdd(QuantInputOutputLayer, Module):

    def __init__(
            self,
            input_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            output_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            tie_input_output_quant = False,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        Module.__init__(self)
        QuantInputOutputLayer.__init__(
            self,
            input_quant,
            output_quant,
            tie_input_output_quant,
            return_quant_tensor,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(
            self,
            input: Union[Tensor, QuantTensor],
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
            self,
            input_quant,
            output_quant,
            tie_input_output_quant,
            return_quant_tensor,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(
            self,
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



