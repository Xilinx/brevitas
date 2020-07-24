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

from typing import Optional, Union
from typing_extensions import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.nn import Identity

from dependencies import Injector

from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector, QuantProxyProtocol


@runtime_checkable
class ActQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: Union[Tensor, QuantTensor]) -> QuantTensor:
        ...


@runtime_checkable
class AccQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: QuantTensor) -> QuantTensor:
        ...


class FusedActivationQuantProxy(torch.jit.ScriptModule):

    def __init__(self, activation_impl, tensor_quant):
        super(FusedActivationQuantProxy, self).__init__()
        self.activation_impl = activation_impl
        self.tensor_quant = tensor_quant

    @torch.jit.script_method
    def forward(self, x):
        x = self.activation_impl(x)
        x, output_scale, output_bit_width = self.tensor_quant(x)
        return x, output_scale, output_bit_width


class ActQuantProxyFromInjector(QuantProxyFromInjector, ActQuantProxyProtocol):

    def __init__(self, act_quant_injector: Injector):
        super(ActQuantProxyFromInjector, self).__init__(act_quant_injector)
        tensor_quant = act_quant_injector.tensor_quant
        act_impl = act_quant_injector.act_impl
        self.is_quant_enabled = tensor_quant is not None
        self.is_act_enabled = act_impl is not None
        if self.is_act_enabled and self.is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                act_impl, tensor_quant)
        elif self.is_act_enabled and not self.is_quant_enabled:
            self.fused_activation_quant_proxy = act_impl
        elif not self.is_act_enabled and self.is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                Identity(), tensor_quant)
        else:
            self.fused_activation_quant_proxy = None

    def scale(self):
        current_status = self.training
        self.eval()  # get eval time scale
        scale = self.__call__(self._zero_hw_sentinel()).scale
        self.train(current_status)
        return scale

    def bit_width(self):
        scale = self.__call__(self._zero_hw_sentinel()).bit_width
        return scale

    def forward(self, x: Union[Tensor, QuantTensor]) -> QuantTensor:
        if self.is_act_enabled or self.is_quant_enabled:
            if isinstance(x, QuantTensor):
                x = x.value
            x = self.fused_activation_quant_proxy(x)
            return QuantTensor(*x, signed=self.is_signed)
        else:
            if isinstance(x, QuantTensor):  # passthrough
                return x
            else:
                return QuantTensor(x)

    def identity_quant(self):
        return IdentityQuantProxyFromInjector(self.quant_injector.let(act_impl=None))

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):

        # scaling_impl_key = prefix + 'fused_activation_quant_proxy.tensor_quant.scaling_impl'
        # runtime_stats_key = scaling_impl_key + '.runtime_stats'
        # running_stats_key = scaling_impl_key + '.runtime_stats.running_stats'
        # scaling_parameter_key = scaling_impl_key + '.value'
        # scaling_affine_weight_key = prefix + '.stats_scaling_impl.affine_rescaling.affine_weight'
        # scaling_affine_bias_key = prefix + '.stats_scaling_impl.affine_rescaling.affine_bias'
        #
        # if not isinstance(self.fused_activation_quant_proxy.tensor_quant, IdentityQuant) and \
        #         self.activation_quant_config.scaling_impl_type == ScalingImplType.PARAMETER:
        #     scaling_impl = self.fused_activation_quant_proxy.tensor_quant.scaling_impl
        #
        #     # If it's retrained directly from statistics, i.e. there isn't a preexisting parameter
        #     if running_stats_key in state_dict and not scaling_parameter_key in state_dict:
        #         scaling_init = state_dict[running_stats_key]
        #         if scaling_affine_weight_key in state_dict:
        #             scaling_init *= state_dict[scaling_affine_weight_key]
        #         if scaling_affine_bias_key in state_dict:
        #             scaling_init += state_dict[scaling_affine_bias_key]
        #
        #         scaling_init = scaling_init.abs()
        #
        #         # Preprocess scaling init, which is always in FP range, based on current value restrictions
        #         restrict_value_type = scaling_impl.restrict_value.restrict_value_type
        #         restrict_value_init_op = scaling_impl.restrict_value.restrict_value_op(restrict_value_type,
        #                                                                                RestrictValueOpImplType.TORCH_FN)
        #         scaling_init = restrict_value_init_op(scaling_init)
        #
        #         # Put scaling init in place in the dict for parameter
        #         if self.activation_quant_config.scaling_impl_type == ScalingImplType.PARAMETER:
        #             state_dict[scaling_parameter_key] = scaling_init
        #
        #     # Get rid of statistics after using them or in case there is already a parameter
        #     for k in list(state_dict.keys()):
        #         if k.startswith(runtime_stats_key):
        #             del state_dict[k]

        # Go on with dict restoring
        super(ActQuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class IdentityQuantProxyFromInjector(ActQuantProxyFromInjector, ActQuantProxyProtocol):

    def __init__(
            self,
            act_quant_injector: Injector):
        assert act_quant_injector.act_impl is None
        super(IdentityQuantProxyFromInjector, self).__init__(act_quant_injector)

    def identity_quant_proxy(self):
        return self


class ClampQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self, clamp_quant_injector: Injector):
        super(ClampQuantProxyFromInjector, self).__init__(clamp_quant_injector)
        tensor_quant = clamp_quant_injector.tensor_quant
        self.is_quant_enabled = tensor_quant is not None
        self.tensor_quant = tensor_quant


    @property
    def is_narrow_range(self):
        return NotImplementedError

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            out_value, out_scale, out_bit_width = self.tensor_quant(x.value, x.scale, x.bit_width)
            return QuantTensor(out_value, out_scale, out_bit_width, self.is_signed)
        return x


class TruncQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self, trunc_quant_injector: Injector):
        super(TruncQuantProxyFromInjector, self).__init__(trunc_quant_injector)
        lsb_trunc_bit_width_impl = trunc_quant_injector.lsb_trunc_bit_width_impl
        tensor_quant = trunc_quant_injector.tensor_quant
        self.is_quant_enabled = lsb_trunc_bit_width_impl is not None and tensor_quant is not None
        self.lsb_trunc_bit_width_impl = lsb_trunc_bit_width_impl
        self.tensor_quant = tensor_quant

    @property
    def is_narrow_range(self):
        return NotImplementedError

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            cleaned_up_value = round_ste(x.value / x.scale.detach()) * x.scale.detach()
            x = x.set(value=cleaned_up_value)  # clean up accumulated floating point errors
            trunc_bit_width = self.lsb_trunc_bit_width_impl(x.bit_width)
            trunc_scale = 2.0 ** trunc_bit_width
            output_scale = trunc_scale * x.scale
            if self.training:
                x, output_scale, x_bit_width = self.tensor_quant(x.value, output_scale, x.bit_width)
            else:  # avoid fp errors at inference time
                x_bit_width = x.bit_width
                x = round_ste(x.value / x.scale)
                x = x / trunc_scale
                x = self.tensor_quant.int_quant.float_to_int_impl(x)
                x = x * output_scale
            x = x / trunc_scale
            output_scale = output_scale / trunc_scale  # output_scale == input_scale
            output_bit_width = x_bit_width - trunc_bit_width
            return QuantTensor(x, output_scale, output_bit_width, self.is_signed)
        else:
            return x
