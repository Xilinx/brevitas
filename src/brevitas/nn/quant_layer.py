# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module

from brevitas.quant_tensor import QuantTensor
from brevitas.utils.torch_utils import compute_channel_view_shape

from .mixin import *
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
        return self.input_quant.is_quant_enabled or self.act_quant.is_quant_enabled

    def forward(self, input: Union[Tensor, QuantTensor]):
        input = self.unpack_input(input)
        quant_input = self.input_quant(input)
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(quant_input)
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
            if self.input_quant.is_quant_enabled and self.act_quant.is_quant_enabled:
                raise RuntimeError("Enable only input or output quant with tie_input_output=True")
            if self.input_quant.is_quant_enabled:
                self.output_quant = self.input_quant
            if self.act_quant.is_quant_enabled:
                self.input_quant = self.output_quant

    @property
    def requires_export_handler(self):
        return self.input_quant.is_quant_enabled or self.act_quant.is_quant_enabled


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
        self._quant_load_model_mode = False

    @abstractmethod
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        pass

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @property
    def requires_export_handler(self):
        return (
            self.input_quant.is_quant_enabled or self.weight_quant.is_quant_enabled or
            self.bias_quant.is_quant_enabled or self.output_quant.is_quant_enabled)

    @property
    def per_elem_ops(self):  # optional, so concrete impl + error if not overridden
        raise NotImplementedError

    def merge_bn_in(self, bn):
        merge_bn(self, bn, output_channel_dim=self.output_channel_dim)

    def forward_impl(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:

        inp = self.unpack_input(inp)

        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(inp)
            self._set_global_is_quant_layer(False)
            return out

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight(quant_input)

        compute_output_quant_tensor = isinstance(quant_input, QuantTensor) and isinstance(
            quant_weight, QuantTensor)
        if not (compute_output_quant_tensor or
                self.output_quant.is_quant_enabled) and self.return_quant_tensor:
            raise RuntimeError("QuantLayer is not correctly configured")

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
        else:
            quant_bias = None
        output_tensor = self.inner_forward_impl(quant_input, quant_weight, quant_bias)

        quant_output = self.output_quant(output_tensor)
        return self.pack_output(quant_output)

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        bias_key = prefix + 'bias'
        # If the state dict has a bias and the module does not, bias correction was used
        # We add a bias module to prevent failing during the load of the state dict
        if (bias_key in state_dict) and (self.bias is None) and self._quant_load_model_mode:
            self.register_parameter(
                'bias',
                torch.nn.Parameter(
                    torch.zeros(
                        self.out_channels, device=self.weight.device, dtype=self.weight.dtype)))
        super(QuantWeightBiasInputOutputLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
