# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.proxy.parameter_quant import BiasQuantProxyProtocol
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol
from brevitas.quant import NoneBiasQuant
from brevitas.quant import NoneWeightQuant
from brevitas.quant_tensor import QuantTensor

from .base import QuantProxyMixin

WeightQuantType = Union[WeightQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]
BiasQuantType = Union[BiasQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class QuantWeightMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, weight_quant: Optional[WeightQuantType], **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=weight_quant,
            proxy_protocol=WeightQuantProxyProtocol,
            none_quant_injector=NoneWeightQuant,
            kwargs_prefix='weight_',
            proxy_prefix='weight_',
            **kwargs)
        self._cached_sub_tensor_slice_list_modules = None

    @property
    @abstractmethod
    def output_channel_dim(self) -> int:
        pass

    def quant_weight(
            self,
            quant_input: Optional[QuantTensor] = None,
            subtensor_slice_list: List[Optional[Tuple[int, int]]] = None):
        weights_to_quantize = self.weight
        if not self.weight_quant.is_quant_enabled and hasattr(self, 'weight_orig'):
            weights_to_quantize = self.weight_orig.to(self.weight.device)
        if subtensor_slice_list is not None:
            # prepare the quantizer for a subtensor input, if any modifications are required
            # we set a list of tuples rather than a list of slices so that it's jit friendly
            # slices generation is handled by each module internally

            # we cache which modules require the attribute
            if self._cached_sub_tensor_slice_list_modules is not None:
                for m in self._cached_sub_tensor_slice_list_modules:
                    m.subtensor_slice_list = subtensor_slice_list
            else:
                self._cached_sub_tensor_slice_list_modules = []
                for m in self.weight_quant.modules():
                    if hasattr(m, 'subtensor_slice_list'):
                        self._cached_sub_tensor_slice_list_modules.append(m)
                        m.subtensor_slice_list = subtensor_slice_list
            # generate slices for the weight tensor based on the list passed in
            weight_slice_tuple = tuple(
                slice(*s) if s is not None else slice(s) for s in subtensor_slice_list)
        else:
            weight_slice_tuple = slice(None)
        if self.weight_quant.requires_quant_input:
            out = self.weight_quant(weights_to_quantize[weight_slice_tuple], quant_input)
        else:
            out = self.weight_quant(weights_to_quantize[weight_slice_tuple])
        if subtensor_slice_list is not None:
            # Restore the quantizer behaviour to full tensor quantization
            # The modules to slice should have been cached already at this point
            assert self._cached_sub_tensor_slice_list_modules is not None, "Missing cache of modules to slice."
            for m in self._cached_sub_tensor_slice_list_modules:
                m.subtensor_slice_list = [None]
        return out

    def register_parameter(self, name, value):
        super(QuantWeightMixin, self).register_parameter(name, value)
        if hasattr(self, 'weight_quant') and name == 'weight':
            self.weight_quant.init_tensor_quant()


class QuantBiasMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            bias_quant: Optional[BiasQuantType],
            cache_inference_bias: bool = False,
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=bias_quant,
            proxy_protocol=BiasQuantProxyProtocol,
            none_quant_injector=NoneBiasQuant,
            kwargs_prefix='bias_',
            proxy_prefix='bias_',
            **kwargs)

    def quant_bias(self):
        if self.bias is None:
            return None
        quant_bias = self.bias_quant(self.bias)
        return quant_bias

    def register_parameter(self, name, value):
        super(QuantBiasMixin, self).register_parameter(name, value)
        if hasattr(self, 'bias_quant') and name == 'bias':
            self.bias_quant.init_tensor_quant()
            self.bias_quant.to(self.bias.device)
