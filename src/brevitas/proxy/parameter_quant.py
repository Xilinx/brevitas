# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Union
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas import config
from brevitas.function import max_int
from brevitas.inject import BaseInjector as Injector
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIO
from brevitas.utils.torch_utils import compute_channel_view_shape

from .quant_proxy import QuantProxyFromInjector
from .quant_proxy import QuantProxyProtocol

__all__ = [
    'WeightQuantProxyFromInjector',
    'WeightQuantProxyFromInjectorBase',
    'BiasQuantProxyFromInjector',
    'BiasQuantProxyFromInjectorBase',
    'WeightQuantProxyProtocol',
    'BiasQuantProxyProtocol']


@runtime_checkable
class WeightQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        ...


@runtime_checkable
class BiasQuantProxyProtocol(QuantProxyProtocol, Protocol):
    requires_input_scale: bool

    def forward(
            self, x: Tensor, input_scale: Optional[Tensor],
            input_bit_width: Optional[Tensor]) -> QuantTensor:
        ...


class ParameterQuantProxyFromInjector(QuantProxyFromInjector):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def tracked_parameter_list(self):
        pass

    def init_tensor_quant(self, preserve_state_dict=False):
        param_list = self.tracked_parameter_list

        # params might not be there yet, e.g. bias before merging
        if param_list:
            if preserve_state_dict:
                reinit_on_state_dict = config.REINIT_ON_STATE_DICT_LOAD
                ignore_missing_key = config.IGNORE_MISSING_KEYS
                config.REINIT_ON_STATE_DICT_LOAD = False
                config.IGNORE_MISSING_KEYS = True
                state_dict = self.state_dict()
            self.quant_injector = self.quant_injector.let(tracked_parameter_list=param_list)
            super(ParameterQuantProxyFromInjector, self).init_tensor_quant()
            if preserve_state_dict:
                self.load_state_dict(state_dict)
                config.IGNORE_MISSING_KEYS = ignore_missing_key
                config.REINIT_ON_STATE_DICT_LOAD = reinit_on_state_dict

    def max_uint_value(self, bit_width):
        return max_int(False, self.is_narrow_range, bit_width)


class WeightQuantProxyFromInjectorBase(ParameterQuantProxyFromInjector,
                                       WeightQuantProxyProtocol,
                                       ABC):

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    @property
    def requires_quant_input(self):
        return False


class BiasQuantProxyFromInjectorBase(ParameterQuantProxyFromInjector, BiasQuantProxyProtocol, ABC):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._cached_bias = None
        self.cache_inference_quant_bias = False

    @property
    def tracked_parameter_list(self):
        return [m.bias for m in self.tracked_module_list if m.bias is not None]

    @property
    def requires_input_scale(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_scale
        else:
            return False

    def get_cached(self, attr):
        if self._cached_bias is None:
            warn(
                "No quant bias cache found, set cache_inference_quant_bias=True and run an "
                "inference pass first")
            return None
        if self.training:
            warn("Cached quant bias scale is being used in training mode.")
        return getattr(self._cached_bias, attr)


class WeightQuantProxyFromInjector(WeightQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._cached_weight = None
        self.cache_inference_quant_weight = False

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    @property
    def requires_quant_input(self):
        return False

    def scale(self):
        if not self.is_quant_enabled:
            return None
        scale = self.__call__(self.tracked_parameter_list[0]).scale
        return scale

    def zero_point(self):
        if not self.is_quant_enabled:
            return None
        zero_point = self.__call__(self.tracked_parameter_list[0]).zero_point
        return zero_point

    def bit_width(self):
        if not self.is_quant_enabled:
            return None
        bit_width = self.__call__(self.tracked_parameter_list[0]).bit_width
        return bit_width

    def forward(self, x: torch.Tensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            if self._cached_weight is not None:
                out = IntQuantTensor(
                    self._cached_weight.quant_tensor.value,
                    self._cached_weight.scale,
                    self._cached_weight.zero_point,
                    self._cached_weight.bit_width,
                    self.is_signed,
                    self.training)
            else:
                impl = self.export_handler if self.export_mode else self.tensor_quant
                out, scale, zero_point, bit_width = impl(x)
                out = IntQuantTensor(
                    out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            out = x
        if isinstance(
                out, IntQuantTensor
        ) and not self.training and self.cache_inference_quant_weight and self._cached_weight is None:
            self._cached_weight = _CachedIO(out.detach(), metadata_only=False)
        return out


class DecoupledWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    def pre_scale(self):
        if not self.is_quant_enabled:
            return None
        output_tuple = self.tensor_quant(self.tracked_parameter_list[0])
        out, scale, zero_point, bit_width, pre_scale, pre_zero_point = output_tuple
        return pre_scale

    def pre_zero_point(self):
        if not self.is_quant_enabled:
            return None
        output_tuple = self.tensor_quant(self.tracked_parameter_list[0])
        out, scale, zero_point, bit_width, pre_scale, pre_zero_point = output_tuple
        return pre_zero_point

    def forward(self, x: torch.Tensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width, pre_scale, pre_zero_point = impl(x)
            return IntQuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return x


class DecoupledWeightQuantWithInputProxyFromInjector(DecoupledWeightQuantProxyFromInjector):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        # Necessary for export
        self._cached_act = None
        self.cache_inference_quant_act = False
        self.cache_quant_io_metadata_only = True

    @property
    def requires_quant_input(self):
        return True

    def pre_scale(self):
        raise NotImplementedError

    def pre_zero_point(self):
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        quant_input: Optional[Union[Tensor,
                                    IntQuantTensor]] = None) -> Union[Tensor, IntQuantTensor]:
        if isinstance(quant_input,
                      IntQuantTensor) and not self.training and self.cache_inference_quant_act:
            cached_inp = _CachedIO(quant_input.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_inp

        if self.is_quant_enabled:
            if quant_input is None:
                assert self._cached_act is not None, "No cached quant input found. Enable caching and perform a forward pass"
                quant_input = self._cached_act
            else:
                assert isinstance(quant_input, IntQuantTensor), "Input must be quantized"

            input_bit_width = quant_input.bit_width
            input_is_signed = quant_input.signed

            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width, pre_scale, pre_zero_point = impl(x, input_bit_width, input_is_signed)
            return IntQuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return x


class BiasQuantProxyFromInjector(BiasQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._cached_bias = None
        self.cache_inference_quant_bias = False

    @property
    def tracked_parameter_list(self):
        return [m.bias for m in self.tracked_module_list if m.bias is not None]

    @property
    def requires_input_scale(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_scale
        else:
            return False

    def get_cached(self, attr):
        if self._cached_bias is None:
            warn(
                "No quant bias cache found, set cache_inference_quant_bias=True and run an "
                "inference pass first")
            return None
        if self.training:
            warn("Cached quant bias scale is being used in training mode.")
        return getattr(self._cached_bias, attr)

    def scale(self):
        if not self.is_quant_enabled:
            return None
        if self.requires_input_scale:
            cache = self.get_cached('scale')
            return cache
        zhs = self._zero_hw_sentinel()
        scale = self.__call__(self.tracked_parameter_list[0], zhs).scale
        return scale

    def zero_point(self):
        if not self.is_quant_enabled:
            return None
        zhs = self._zero_hw_sentinel()
        zero_point = self.__call__(self.tracked_parameter_list[0], zhs).zero_point
        return zero_point

    def bit_width(self):
        if not self.is_quant_enabled:
            return None
        zhs = self._zero_hw_sentinel()
        bit_width = self.__call__(self.tracked_parameter_list[0], zhs).bit_width
        return bit_width

    def quant_output_scale_impl(
            self, input: IntQuantTensor, weight: IntQuantTensor, module: torch.nn.Module) -> Tensor:
        channel_dim = -1 if isinstance(module, torch.nn.Linear) else 1
        output_scale_shape = compute_channel_view_shape(input, channel_dim=channel_dim)
        output_scale = weight.scale.view(output_scale_shape)
        output_scale = output_scale * input.scale.view(output_scale_shape)
        return output_scale

    def compute_bias_scale(
            self,
            input: Optional[Union[Tensor, IntQuantTensor]],
            weight: Optional[Union[Tensor, IntQuantTensor]]) -> Optional[Tensor]:
        if not self.requires_input_scale:
            return None
        if not isinstance(input, IntQuantTensor) or not isinstance(weight, IntQuantTensor):
            return None
        if len(self.tracked_module_list) > 1:
            if not all(
                [type[self.tracked_module_list[0]] == type[x] for x in self.tracked_module_list]):
                raise RuntimeError(
                    "Bias quantizer shared across different type of layers with external scale is not supported."
                )
        scale = self.quant_output_scale_impl(input, weight, self.tracked_module_list[0])
        return scale

    def forward(
            self,
            x: Tensor,
            input: Optional[Union[Tensor, IntQuantTensor]] = None,
            weight: Optional[Union[Tensor,
                                   IntQuantTensor]] = None) -> Union[Tensor, IntQuantTensor]:
        out = x
        input_scale = self.compute_bias_scale(input, weight)
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            if self.requires_input_scale and input_scale is None:
                input_scale = self.scale()
                if input_scale is None:
                    raise RuntimeError("Input scale required")

            if self.requires_input_scale:
                input_scale = input_scale.view(-1)
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale)
            else:
                out, out_scale, out_zp, out_bit_width = impl(x)

            out = IntQuantTensor(
                out, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        else:
            out = x
        if isinstance(out,
                      IntQuantTensor) and not self.training and self.cache_inference_quant_bias:
            cached_bias = _CachedIO(out.detach(), metadata_only=False)
            self._cached_bias = cached_bias
        return out
