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

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from brevitas.quant_tensor import QuantTensor
from .mixin.base import _CachedIO
from .mixin import *

from .utils import rename_state_dict_by_prefix, compute_channel_view_shape
from .utils import merge_bn


class QuantNonLinearActLayer(
    QuantNonLinearActMixin,
    QuantInputMixin,
    QuantLayerMixin,
    Module):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Optional[Type[Module]],
            passthrough_act: bool,
            input_quant: Optional[ActQuantType],
            act_quant: Optional[ActQuantType],
            return_quant_tensor: bool,
            **kwargs):
        Module.__init__(self)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantInputMixin.__init__(self, input_quant, **kwargs)
        QuantNonLinearActMixin.__init__(
            self,
            act_impl,
            passthrough_act,
            act_quant,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return self.is_input_quant_enabled or self.is_act_quant_enabled

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self.is_input_quant_enabled:
            return self.input_quant.is_signed
        elif self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    @property
    def is_quant_act_signed(self):
        if self.is_act_quant_enabled:
            return self.act_quant.is_signed
        elif self._cached_out is not None:
            return self._cached_out.signed
        else:  # raise exception instead of returning None since None would be falsey
            raise RuntimeError("Enable quant output caching or act quantization")

    @property
    def is_quant_output_signed(self):  # overrides from QuantLayerMixin
        return self.is_quant_act_signed

    def quant_input_scale(self):
        if self.is_input_quant_enabled:
            return self.input_quant.scale()
        elif self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_act_scale(self):
        if self.is_act_quant_enabled:
            return self.act_quant.scale()
        elif self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_scale(self):  # overrides from QuantLayerMixin
        return self.quant_act_scale()

    def quant_input_zero_point(self):
        if self.is_input_quant_enabled:
            return self.input_quant.zero_point()
        elif self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_act_zero_point(self):
        if self.is_act_quant_enabled:
            return self.act_quant.zero_point()
        elif self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_zero_point(self):  # overrides from QuantLayerMixin
        return self.quant_act_zero_point()

    def quant_input_bit_width(self):
        if self.is_input_quant_enabled:
            return self.input_quant.bit_width()
        elif self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    def quant_act_bit_width(self):
        if self.is_act_quant_enabled:
            return self.act_quant.bit_width()
        elif self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def quant_output_bit_width(self):   # overrides from QuantLayerMixin
        return self.quant_act_bit_width()

    def forward(self, input: Union[Tensor, QuantTensor]):
        input = self.unpack_input(input)
        quant_input = self.input_quant(input)
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(quant_input.value)
            self._set_global_is_quant_layer(False)
            return out
        out = self.act_quant(quant_input)
        out = self.pack_output(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # for retrocompatibility
        rename_state_dict_by_prefix(prefix + 'act_quant_proxy', prefix + 'act_quant', state_dict)
        super(QuantNonLinearActLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantInputOutputLayer(
        QuantOutputMixin,
        QuantInputMixin,
        QuantLayerMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            input_quant: Optional[ActQuantType],
            output_quant: Optional[ActQuantType],
            tie_input_output_quant: bool,
            return_quant_tensor: bool,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantInputMixin.__init__(self, input_quant, **kwargs)
        QuantOutputMixin.__init__(self, output_quant, **kwargs)
        # we have to account for quantization being enabled through kwargs
        if tie_input_output_quant:
            if self.is_input_quant_enabled and self.is_output_quant_enabled:
                raise RuntimeError("Enable only input or output quant with tie_input_output=True")
            if self.is_input_quant_enabled:
                self.output_quant = self.input_quant
            if self.is_output_quant_enabled:
                self.input_quant = self.output_quant

    @property
    def requires_export_handler(self):
        return self.is_input_quant_enabled or self.is_output_quant_enabled

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self.is_input_quant_enabled:
            return self.input_quant.is_signed
        elif self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    @property
    def is_quant_output_signed(self)  -> Optional[bool]:  # tri-valued logic output:
        if self.is_output_quant_enabled:
            return self.output_quant.is_signed
        elif self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_input_scale(self):
        if self.is_input_quant_enabled:
            return self.input_quant.scale()
        elif self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_output_scale(self):
        if self.is_output_quant_enabled:
            return self.output_quant.scale()
        elif self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self.is_input_quant_enabled:
            return self.input_quant.zero_point()
        elif self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_output_zero_point(self):
        if self.is_output_quant_enabled:
            return self.output_quant.zero_point()
        elif self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self.is_input_quant_enabled:
            return self.input_quant.bit_width()
        elif self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    def quant_output_bit_width(self):
        if self.is_output_quant_enabled:
            return self.output_quant.bit_width()
        elif self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None


class QuantWeightBiasInputOutputLayer(
        QuantBiasMixin,
        QuantWeightMixin,
        QuantInputOutputLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant: Optional[WeightQuantType],
            bias_quant: Optional[BiasQuantType],
            input_quant: Optional[ActQuantType],
            output_quant: Optional[ActQuantType],
            return_quant_tensor: bool,
            **kwargs):
        QuantInputOutputLayer.__init__(
            self,
            input_quant,
            output_quant,
            tie_input_output_quant=False,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        QuantWeightMixin.__init__(self, weight_quant, **kwargs)
        QuantBiasMixin.__init__(self, bias_quant, **kwargs)

    @abstractmethod
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        pass

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @property
    def requires_export_handler(self):
        return (self.is_input_quant_enabled
                or self.is_weight_quant_enabled
                or self.is_bias_quant_enabled
                or self.is_output_quant_enabled)

    @property
    def per_elem_ops(self):  # optional, so concrete impl + error if not overridden
        raise NotImplementedError

    def merge_bn_in(self, bn):
        merge_bn(self, bn, output_channel_dim=self.output_channel_dim)

    def forward_impl(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        output_scale = None
        output_bit_width = None
        output_zero_point = None
        output_signed = None

        inp = self.unpack_input(inp)

        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(inp.value)
            self._set_global_is_quant_layer(False)
            return out

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight()

        if quant_input.bit_width is not None:
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
        if quant_input.scale is not None:
            output_scale_shape = compute_channel_view_shape(inp, channel_dim=1)
            output_scale = quant_weight.scale.view(output_scale_shape)
            output_scale = output_scale * quant_input.scale.view(output_scale_shape)
        if quant_input.signed is not None:
            output_signed = inp.signed or quant_weight.signed

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, output_scale, output_bit_width)
            if not self.training and self.cache_inference_quant_bias:
                self._cached_bias = _CachedIO(quant_bias.detach(), metadata_only=False)

            output_tensor = self.inner_forward_impl(
                quant_input.value, quant_weight.value, quant_bias.value)

            if (output_scale is not None
                    and (quant_bias.scale is None
                         or (quant_bias.scale is not None
                             and quant_bias.scale.data_ptr() != output_scale.data_ptr()))):
                output_zero_point = - quant_bias.value.view(output_scale_shape) / output_scale

            if quant_bias.bit_width is not None and output_bit_width is not None:
                output_bit_width = torch.where(
                    quant_bias.bit_width > output_bit_width, quant_bias.bit_width, output_bit_width)
                output_bit_width = output_bit_width + 1
        else:
            output_tensor = self.inner_forward_impl(quant_input.value, quant_weight.value, None)

        if self.return_quant_tensor and not self.is_output_quant_enabled:
            if (quant_input.zero_point is not None
                    and ((quant_input.zero_point != 0.0).any()
                         or (quant_weight.zero_point != 0.0).any())):
                raise RuntimeError("Computing zero point of output accumulator not supported yet.")
            elif quant_input.zero_point is not None and output_zero_point is None:
                output_zero_point = quant_input.zero_point

        quant_output = QuantTensor(
            value=output_tensor,
            scale=output_scale,
            zero_point=output_zero_point,
            bit_width=output_bit_width,
            signed=output_signed,
            training=self.training)
        quant_output = self.output_quant(quant_output)
        return self.pack_output(quant_output)



