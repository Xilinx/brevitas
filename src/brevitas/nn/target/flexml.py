# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

import torch
from torch import nn
from torch import Tensor

from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
import brevitas.nn as qnn
from brevitas.nn.mixin import QuantLayerMixin
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant import Uint8ActPerTensorFixedPoint
from brevitas.quant_tensor import QuantTensor


class FlexMLQuantLeakyReLU(nn.Module):

    def __init__(
        self,
        negative_slope,
        alpha_quant=qnn.QuantIdentity(Uint8ActPerTensorFixedPoint, bit_width=16),
        input_quant=qnn.QuantIdentity(
            Int8ActPerTensorFixedPoint, bit_width=16, scaling_stats_momentum=None),
        output_quant=qnn.QuantIdentity(Int8ActPerTensorFixedPoint, return_quant_tensor=True)):
        super(FlexMLQuantLeakyReLU, self).__init__()
        self.alpha_quant = alpha_quant
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.negative_slope = StatelessBuffer(torch.tensor(negative_slope))

    def forward(self, inp):
        quant_inp = self.input_quant(inp)
        quant_alpha = self.alpha_quant(self.negative_slope())
        quant_alpha_out = self.input_quant(quant_inp * quant_alpha)
        out = torch.max(quant_inp, quant_alpha_out)
        out = self.output_quant(out)
        return out

    @property
    def act_quant(self):
        return self.output_quant.act_quant

    @property
    def is_quant_act_signed(self):
        return self.output_quant.is_quant_act_signed


class FlexMLQuantSwish(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFixedPoint,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.SiLU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class FlexMLQuantHardsigmoid(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFixedPoint,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Hardsigmoid,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class FlexMLQuantHardswish(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFixedPoint,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Hardswish,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class FlexMLQuantAvgPool2d(QuantLayerMixin, nn.AvgPool2d):

    class Int16QuantAvgPoolDivQuant(Int8ActPerTensorFixedPoint):
        bit_width = 8  # div_factor is a 16b in the backend so it could go up to 16b in principle
        scaling_stats_op = 'max'
        float_to_int_impl_type = 'round'  # round to nearest ties to even

    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            ceil_mode=False,
            div_quant=Int16QuantAvgPoolDivQuant,
            return_quant_tensor=True) -> None:
        nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=True)  # fixed to True to make sure the kernel size is consistent
        QuantLayerMixin.__init__(self, return_quant_tensor)
        div_quant = qnn.QuantIdentity(div_quant, return_quant_tensor=True)
        quantized_div = div_quant(torch.tensor(1. / self._avg_scaling))
        self.quantized_div_int_value = quantized_div.int().item()
        self.quantized_div_scale = quantized_div.scale.item()
        self.rescaling_const = quantized_div.value.item() * self._avg_scaling

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    @property
    def _avg_scaling(self):
        if isinstance(self.kernel_size, tuple):
            return self.kernel_size[0] * self.kernel_size[1]
        else:
            return self.kernel_size * self.kernel_size

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        x = x.set(value=super(FlexMLQuantAvgPool2d, self).forward(x.value) * self.rescaling_const)
        if x.scale is not None:
            x = x.set(scale=x.scale * self.quantized_div_scale)
        if x.bit_width is not None:
            x = x.set(bit_width=self.max_acc_bit_width(x.bit_width))
        return self.pack_output(x)

    def max_acc_bit_width(self, input_bit_width):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_uint_output = max_uint_input * self._avg_scaling * self.quantized_div_int_value
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
