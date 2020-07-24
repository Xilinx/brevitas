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

from abc import ABCMeta, abstractmethod
from typing import Optional, Type, Union, Callable

from dependencies import Injector
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from brevitas.quant_tensor import QuantTensor
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol, BiasQuantProxyProtocol
from brevitas.proxy.runtime_quant import ActQuantProxyProtocol
from brevitas.proxy.config import update_weight_quant_injector as default_update_wqi
from brevitas.proxy.config import update_bias_quant_injector as default_update_bqi
from brevitas.proxy.config import update_act_quant_injector as default_update_aqi
from .mixin import *

from .utils import mul_add_from_bn


class DefaultWeightScalingInjector(Injector):
    scaling_impl_type = 'STATS'
    restrict_scaling_type = 'FP'
    scaling_stats_op = 'MAX'
    scaling_per_output_channel = False
    scaling_min_val = 2.0 ** (-16)


class DefaultWeightQuantInjector(DefaultWeightScalingInjector):
    quant_type = 'INT'
    bit_width_impl_type = 'CONST'
    narrow_range = True
    signed = True
    bit_width = 8


class DefaultActQuantInjector(Injector):
    quant_type = 'INT'
    bit_width_impl_type = 'CONST'
    bit_width = 8
    scaling_impl_type = 'PARAMETER'
    restrict_scaling_type = 'LOG_FP'
    scaling_per_output_channel = False
    scaling_min_val = 2.0 ** (-16)


class DefaultTruncQuantInjector(Injector):
    quant_type = 'INT'
    lsb_trunc_bit_width_impl_type = 'CONST'
    narrow_range = False
    min_overall_bit_width = 2  # TODO deprecate
    max_overall_bit_width = 32  # TODO deprecate


class DefaultSignedActQuantInjector(DefaultActQuantInjector):
    signed = True
    narrow_range = False


class DefaultUnsignedActQuantInjector(DefaultActQuantInjector):
    signed = False
    narrow_range = False
    min_val = 0.0


class DefaultUnitarySignedActQuantInjector(DefaultSignedActQuantInjector):
    min_val = -1.0
    max_val = 1.0


class DefaultUnitaryUnsignedActQuantInjector(DefaultUnsignedActQuantInjector):
    max_val = 1.0


def _compute_channel_view_shape(tensor: Tensor, channel_dim: int):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)
    

class QuantNonLinearActLayer(QuantNonLinearActMixin, QuantLayerMixin, Module):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Optional[Module],
            act_quant: Union[ActQuantProxyProtocol, Type[Injector]],
            return_quant_tensor: bool,
            update_aqi: Callable = default_update_aqi,
            **kwargs):
        Module.__init__(self)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantNonLinearActMixin.__init__(self, act_impl, act_quant, update_aqi, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(self, inp: Union[Tensor, QuantTensor]):
        inp = self.unpack_input(inp)
        if self.export_mode:  # shortcut execution through the export impl
            return self.export_handler(inp.value)
        out = self.act_quant(inp)
        out = self.pack_output(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = 'act_quant'
        retrocomp_value_key = prefix + 'act_quant_proxy'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(QuantNonLinearActLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantWeightBiasInputOutputLayer(
        QuantOutputMixin,
        QuantInputMixin,
        QuantBiasMixin,
        QuantWeightMixin,
        QuantLayerMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight: Parameter,
            bias: Parameter,
            weight_quant: Union[WeightQuantProxyProtocol, Type[Injector]],
            bias_quant: Union[BiasQuantProxyProtocol, Type[Injector]],
            input_quant: Union[ActQuantProxyProtocol, Type[Injector]],
            output_quant: Union[ActQuantProxyProtocol, Type[Injector]],
            return_quant_tensor: bool,
            update_wqi: Callable = default_update_wqi,
            update_bqi: Callable = default_update_bqi,
            update_iqi: Callable = default_update_aqi,
            update_oqi: Callable = default_update_aqi,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantWeightMixin.__init__(self, weight, weight_quant, update_wqi, **kwargs)
        QuantBiasMixin.__init__(self, bias, bias_quant, update_bqi, **kwargs)
        QuantInputMixin.__init__(self, input_quant, update_iqi, **kwargs)
        QuantOutputMixin.__init__(self, output_quant, update_oqi, **kwargs)

    @property
    def per_elem_ops(self):  # optional, so concrete impl + error if not overridden
        raise NotImplementedError

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @abstractmethod
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        pass

    def merge_bn_in(self, bn, affine_only):
        if affine_only and not bn.affine:
            raise RuntimeError("Affine-only merging requires BN to have affine scaling enabled.")
        else:
            out = mul_add_from_bn(
                bn_mean=bn.running_mean,
                bn_var=bn.running_var,
                bn_eps=bn.eps,
                bn_weight=bn.weight.data.clone(),
                bn_bias=bn.bias.data.clone(),
                affine_only=affine_only)
            mul_factor, add_factor = out
            out_ch_weight_shape = _compute_channel_view_shape(self.weight, self.output_channel_dim)
            self.weight.data *= mul_factor.view(out_ch_weight_shape)
            if self.bias is not None:
                out_ch_bias_shape = _compute_channel_view_shape(self.bias, self.output_channel_dim)
                self.bias.data += add_factor.view(out_ch_bias_shape)
            else:
                self.bias = Parameter(add_factor)

    def forward_impl(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        output_scale = None
        output_bit_width = None

        inp = self.unpack_input(inp)

        if self.export_mode:  # shortcut execution through the export impl
            return self.export_handler(inp.value)

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight()

        if quant_input.bit_width is not None:
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
        if quant_input.scale is not None:
            output_scale_shape = _compute_channel_view_shape(inp, channel_dim=1)
            output_scale = quant_weight.scale.view(output_scale_shape)
            output_scale = output_scale * quant_input.scale.view(output_scale_shape)

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, output_scale, output_bit_width)
            output_tensor = self.inner_forward_impl(
                quant_input.value, quant_weight.value, quant_bias.value)
            if quant_bias.bit_width is not None:
                output_bit_width = torch.where(
                    quant_bias.bit_width > output_bit_width, quant_bias.bit_width, output_bit_width)
                output_bit_width = output_bit_width + 1
        else:
            output_tensor = self.inner_forward_impl(quant_input.value, quant_weight.value, None)

        quant_output = QuantTensor(output_tensor, output_scale, output_bit_width, signed=True)
        quant_output = self.output_quant(quant_output)
        return self.pack_output(quant_output)









