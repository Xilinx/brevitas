# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor

from .mixin import *
from .mixin.base import _CachedIO
from .utils import compute_channel_view_shape
from .utils import merge_bn
from .utils import rename_state_dict_by_prefix


class QuantNonLinearActLayer(QuantNonLinearActMixin, QuantInputMixin, QuantLayerMixin, Module):
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
        QuantNonLinearActMixin.__init__(self, act_impl, passthrough_act, act_quant, **kwargs)

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
    def is_quant_act_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self.is_act_quant_enabled:
            return self.act_quant.is_signed
        elif self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    @property
    def is_output_quant_enabled(self):
        return self.is_act_quant_enabled

    @property
    def is_quant_output_narrow_range(self):
        return self.is_quant_act_narrow_range

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

    def quant_output_bit_width(self):  # overrides from QuantLayerMixin
        return self.quant_act_bit_width()

    def forward(self, input: Union[Tensor, QuantTensor]):
        input = self.unpack_input(input)
        quant_input = self.input_quant(input)
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(quant_input))
            self._set_global_is_quant_layer(False)
            return out
        out = self.act_quant(quant_input)
        out = self.pack_output(out)
        return out

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        # for retrocompatibility
        rename_state_dict_by_prefix(prefix + 'act_quant_proxy', prefix + 'act_quant', state_dict)
        super(QuantNonLinearActLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantInputOutputLayer(QuantOutputMixin, QuantInputMixin, QuantLayerMixin):
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
    def is_quant_output_signed(self) -> Optional[bool]:  # tri-valued logic output:
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


class QuantWeightBiasInputOutputLayer(QuantBiasMixin, QuantWeightMixin, QuantInputOutputLayer):
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

    def quant_output_scale_impl(
            self, inp: Tensor, quant_input_scale: Tensor, quant_weight_scale: Tensor):
        output_scale_shape = compute_channel_view_shape(inp, channel_dim=1)
        output_scale = quant_weight_scale.view(output_scale_shape)
        output_scale = output_scale * quant_input_scale.view(output_scale_shape)
        return output_scale

    @property
    def requires_export_handler(self):
        return (
            self.is_input_quant_enabled or self.is_weight_quant_enabled or
            self.is_bias_quant_enabled or self.is_output_quant_enabled)

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
            out = self.export_handler(_unpack_quant_tensor(inp))
            self._set_global_is_quant_layer(False)
            return out

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight(quant_input)

        compute_output_quant_tensor = isinstance(quant_input, QuantTensor) and isinstance(
            quant_weight, QuantTensor)
        if not (compute_output_quant_tensor or
                self.is_output_quant_enabled) and self.return_quant_tensor:
            raise RuntimeError("QuantLayer is not correctly configured")

        if isinstance(quant_input, QuantTensor) and isinstance(quant_weight, QuantTensor):
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
            output_scale = self.quant_output_scale_impl(inp, quant_input.scale, quant_weight.scale)
            output_signed = quant_input.signed or quant_weight.signed

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, output_scale)
            if not self.training and self.cache_inference_quant_bias and isinstance(quant_bias,
                                                                                    QuantTensor):

                self._cached_bias = _CachedIO(quant_bias.detach(), metadata_only=False)
            output_tensor = self.inner_forward_impl(
                _unpack_quant_tensor(quant_input),
                _unpack_quant_tensor(quant_weight),
                _unpack_quant_tensor(quant_bias))

            if output_scale is not None:
                if (isinstance(quant_bias, QuantTensor) and
                        quant_bias.scale.data_ptr() != output_scale.data_ptr()) or not isinstance(
                            quant_bias, QuantTensor):
                    channel_dim = -1 if isinstance(self, torch.nn.Linear) else 1
                    output_scale_broadcast_shape = compute_channel_view_shape(
                        inp, channel_dim=channel_dim)
                    output_zero_point = -_unpack_quant_tensor(quant_bias).view(
                        output_scale_broadcast_shape) / output_scale

            if output_bit_width is not None and isinstance(quant_bias, QuantTensor):
                output_bit_width = torch.where(
                    quant_bias.bit_width > output_bit_width, quant_bias.bit_width, output_bit_width)
                output_bit_width = output_bit_width + 1
        else:
            output_tensor = self.inner_forward_impl(
                _unpack_quant_tensor(quant_input), _unpack_quant_tensor(quant_weight), None)

        if not self.is_output_quant_enabled and self.return_quant_tensor:
            if compute_output_quant_tensor:
                if (quant_input.zero_point != 0.0).any() or (quant_weight.zero_point != 0.0).any():
                    raise RuntimeError(
                        "Computing zero point of output accumulator not supported yet.")
                elif output_zero_point is None:
                    output_zero_point = quant_input.zero_point

        elif output_zero_point is None:
            output_zero_point = torch.zeros(1).type_as(output_tensor)

        if compute_output_quant_tensor:
            quant_output = QuantTensor(
                output_tensor,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=self.training)
        else:
            quant_output = output_tensor

        quant_output = self.output_quant(quant_output)
        return self.pack_output(quant_output)
