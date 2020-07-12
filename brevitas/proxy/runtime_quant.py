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

import torch
from torch import Tensor
from torch.nn import Module, Identity

from dependencies import Injector

from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxy


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


class ActQuantProxy(QuantProxy):

    def __init__(
            self,
            act_quant_injector: Injector):
        super(ActQuantProxy, self).__init__()
        tensor_quant = act_quant_injector.tensor_quant
        act_impl = act_quant_injector.act_impl
        quant_enabled = tensor_quant is not None
        act_enabled = act_impl is not None
        self.act_quant_injector = act_quant_injector
        if act_enabled and quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                act_impl, tensor_quant)
        elif act_enabled and not quant_enabled:
            self.fused_activation_quant_proxy = act_impl
        elif not act_enabled and quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                Identity(), tensor_quant)
        else:
            self.fused_activation_quant_proxy = None

    def identity_quant(self):
        return IdentityQuantProxy(self.act_quant_injector.let(act_impl=None))

    def quant_act_scale(self): # TODO
        # if isinstance(self.act_quant.fused_act_quant_proxy.tensor_quant, IdentityQuant):
        #     raise Exception("Can't generate scaling factor without quantization enabled")
        # scaling_impl = self.act_quant_proxy.fused_act_quant_proxy.tensor_quant.scaling_impl
        # current_status = scaling_impl.training
        # scaling_impl.eval()
        # _, out, _ = self.act_quant_proxy(self.act_quant_proxy._zero_hw_sentinel())
        # scaling_impl.train(current_status)
        #return out
        pass

    def forward(self, x: Union[Tensor, QuantTensor]):
        if isinstance(x, QuantTensor) and self.fused_activation_quant_proxy is not None:
            x = self.fused_activation_quant_proxy(x.value)
        else:
            x = self.fused_activation_quant_proxy(x)
        if isinstance(x, tuple): # quantization happened
            return QuantTensor(*x, signed=self.activation_quant_injector.signed)
        elif isinstance(x, QuantTensor):  # x is still the input to the forward, pass it through
            return x
        else:  # only activation_impl was called
            return QuantTensor(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs): # TODO

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
        super(ActQuantProxy, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class IdentityQuantProxy(ActQuantProxy):

    def __init__(
            self,
            act_quant_injector: Injector):
        assert act_quant_injector.act_impl is None
        super(IdentityQuantProxy, self).__init__(act_quant_injector)

    def identity_quant_proxy(self):
        return self


class ClampQuantProxy(QuantProxy):

    def __init__(self,
                 signed: bool,
                 narrow_range: bool,
                 quant_type,
                 ms_bit_width_to_clamp: int,
                 clamp_at_least_init_val: bool,
                 min_overall_bit_width: Optional[int],
                 max_overall_bit_width: Optional[int],
                 msb_clamp_bit_width_impl_type,
                 override_pretrained_bit_width: bool):
        super(ClampQuantProxy, self).__init__()

        if quant_type == QuantType.FP:
            self.tensor_quant = IdentityPrescaledIntQuant()

        elif quant_type == QuantType.INT:
            msb_clamp_bit_width_impl = MsbClampParameterBitWidth(ms_bit_width_to_clamp=ms_bit_width_to_clamp,
                                                                 clamp_at_least_init_val=clamp_at_least_init_val,
                                                                 min_overall_bit_width=min_overall_bit_width,
                                                                 max_overall_bit_width=max_overall_bit_width,
                                                                 bit_width_impl_type=msb_clamp_bit_width_impl_type,
                                                                 override_pretrained=override_pretrained_bit_width)
            tensor_clamp_impl = TensorClamp()
            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.ROUND,
                                              min_val=None)
            tensor_quant_impl = PrescaledRestrictIntQuantWithInputBitWidth
            self.tensor_quant = tensor_quant_impl(signed=signed,
                                                  narrow_range=narrow_range,
                                                  tensor_clamp_impl=tensor_clamp_impl,
                                                  float_to_int_impl=float_to_int_impl,
                                                  msb_clamp_bit_width_impl=msb_clamp_bit_width_impl)
        else:
            raise Exception("Quantization type {} not supported for accumulators.".format(quant_type))

    def forward(self, x, input_scale, input_bit_width):
        x, output_scale, output_bit_width = self.tensor_quant(x, input_scale, input_bit_width)
        return x, output_scale, output_bit_width


class TruncQuantProxy(QuantProxy):

    def __init__(self,
                 signed: bool,
                 quant_type,
                 ls_bit_width_to_trunc: int,
                 trunc_at_least_init_val: bool,
                 min_overall_bit_width: Optional[int],
                 max_overall_bit_width: Optional[int],
                 lsb_trunc_bit_width_impl_type,
                 explicit_rescaling: bool,
                 override_pretrained_bit_width: bool):
        super(TruncQuantProxy, self).__init__()
        self.explicit_rescaling = explicit_rescaling

        if quant_type == QuantType.FP:
            self.lsb_trunc_bit_width_impl = ZeroLsbTruncBitWidth()
            self.tensor_quant = IdentityPrescaledIntQuant()

        elif quant_type == QuantType.INT:
            self.lsb_trunc_bit_width_impl = LsbTruncParameterBitWidth(ls_bit_width_to_trunc=ls_bit_width_to_trunc,
                                                                      trunc_at_least_init_val=trunc_at_least_init_val,
                                                                      min_overall_bit_width=min_overall_bit_width,
                                                                      max_overall_bit_width=max_overall_bit_width,
                                                                      bit_width_impl_type=lsb_trunc_bit_width_impl_type,
                                                                      override_pretrained=override_pretrained_bit_width)
            tensor_clamp_impl = TensorClamp()
            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.FLOOR,
                                              min_val=None)
            msb_clamp_bit_width_impl = IdentityBitWidth()
            self.tensor_quant = PrescaledRestrictIntQuantWithInputBitWidth(narrow_range=False,
                                                                           signed=signed,
                                                                           tensor_clamp_impl=tensor_clamp_impl,
                                                                           msb_clamp_bit_width_impl=msb_clamp_bit_width_impl,
                                                                           float_to_int_impl=float_to_int_impl)

        else:
            raise Exception("Quantization type {} not supported for accumulators.".format(quant_type))

    def forward(self, x, input_scale, input_bit_width):
        x = round_ste(x / input_scale) * input_scale  # clean up fp errors before floor
        trunc_bit_width = self.lsb_trunc_bit_width_impl(input_bit_width)
        trunc_scale = 2.0 ** trunc_bit_width
        output_scale = trunc_scale * input_scale
        x, output_scale, input_bit_width = self.tensor_quant(x, output_scale, input_bit_width)
        if self.explicit_rescaling:
            x = x / trunc_scale  # rescaling is explicit, so the truncation scale stays with x rather with output_scale
            output_scale = output_scale / trunc_scale
        output_bit_width = input_bit_width - trunc_bit_width
        return x, output_scale, output_bit_width
