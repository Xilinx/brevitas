# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas import config
from brevitas import is_dynamo_compiling
from brevitas.core.function_wrapper.misc import Identity
from brevitas.function import max_int
from brevitas.inject import BaseInjector as Injector
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIO
from brevitas.utils.torch_utils import compute_channel_view_shape
from brevitas.utils.torch_utils import is_broadcastable

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

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._cached_weight = None
        self._cache_inference_quant_weight = False
        self.cache_inference_quant_weight_metadata_only = False
        self.cache_class = None  # To be redefined by each class
        self.quant_tensor_class = None  # To be redefined by each class
        self.skip_create_quant_tensor = False

    def compile_quant(self, compile_export=False):
        if compile_export and hasattr(self, 'export_handler') and self.export_handler is not None:
            self.export_handler.inner_forward = torch.compile(
                self.export_handler.inner_forward, dynamic=True, fullgraph=True)
        elif self.tensor_quant is not None:
            # For groupwise weight quantization, we have graph breaks
            fullgraph = not self.is_groupwise
            self.tensor_quant = torch.compile(self.tensor_quant, dynamic=True, fullgraph=fullgraph)

    @property
    def input_view_impl(self):
        if self.tensor_quant is not None:
            return self.tensor_quant.int_quant.input_view_impl
        else:
            return Identity()

    @property
    def cache_inference_quant_weight(self):
        return self._cache_inference_quant_weight

    @cache_inference_quant_weight.setter
    def cache_inference_quant_weight(self, value):
        if not value:
            self._cached_weight = None
        self._cache_inference_quant_weight = value

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    def retrieve_attribute(self, attribute: str):
        if not self.is_quant_enabled:
            return None
        elif self._cached_weight is not None:
            return getattr(self._cached_weight, attribute)
        else:
            out = self.__call__(self.tracked_parameter_list[0])
            return getattr(out, attribute)

    @property
    def requires_quant_input(self):
        return False

    @abstractmethod
    def create_quant_tensor(self, qt_args: Tuple[Any]) -> Union[Tensor, QuantTensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Union[Tensor, QuantTensor]:
        if self.is_quant_enabled:
            # If quant is enabled the priority is:
            # - export mode
            # - quantization flow
            if self.export_mode:
                out = self.export_handler(x)
                if self.skip_create_quant_tensor:
                    out = out[0]
                else:
                    out = self.create_quant_tensor(out)
            else:
                out = self.tensor_quant(x)
                if self.skip_create_quant_tensor:
                    out = out[0]
                else:
                    out = self.create_quant_tensor(out)
                    if not self.training and self.cache_inference_quant_weight and self._cached_weight is None:
                        self._cached_weight = self.cache_class(
                            out.detach(),
                            metadata_only=self.cache_inference_quant_weight_metadata_only)
        else:  # quantization disabled
            out = x
        return out


class BiasQuantProxyFromInjectorBase(ParameterQuantProxyFromInjector, BiasQuantProxyProtocol, ABC):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._cached_bias = None
        self.cache_inference_quant_bias = False
        self.cache_inference_quant_bias_metadata_only = False
        self.requires_input_scale = self.quant_injector.requires_input_scale
        self.skip_create_quant_tensor = False

    @property
    def tracked_parameter_list(self):
        return [m.bias for m in self.tracked_module_list if m.bias is not None]

    def get_cached(self, attr):
        if self._cached_bias is None:
            if not is_dynamo_compiling():
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
        self.cache_class = _CachedIO

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    @property
    def requires_quant_input(self):
        return False

    def scale(self):
        return self.retrieve_attribute('scale')

    def zero_point(self):
        return self.retrieve_attribute('zero_point')

    def bit_width(self):
        return self.retrieve_attribute('bit_width')

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> IntQuantTensor:
        return IntQuantTensor(*qt_args, self.is_signed, self.training)


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

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> IntQuantTensor:
        out, scale, zero_point, bit_width, pre_scale, pre_zero_point = qt_args
        return IntQuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)


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
            if quant_input is None or isinstance(quant_input, Tensor):
                assert self._cached_act is not None, "No cached quant input found. Enable caching and perform a forward pass"
                quant_input = self._cached_act
            else:
                assert isinstance(quant_input, IntQuantTensor), "Input must be quantized"

            input_bit_width = quant_input.bit_width
            input_is_signed = quant_input.signed

            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width, pre_scale, pre_zero_point = impl(x, input_bit_width, input_is_signed)
            if self.skip_create_quant_tensor:
                return out
            return IntQuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return x


class BiasQuantProxyFromInjector(BiasQuantProxyFromInjectorBase):

    def scale(self):
        if not self.is_quant_enabled:
            return None
        if self.requires_input_scale and self.is_quant_enabled:
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

        input_scale_view = input.scale.view(output_scale_shape)
        if not is_broadcastable(output_scale.shape, input_scale_view.shape):
            return None

        output_scale = output_scale * input_scale_view
        return output_scale

    def compute_bias_scale(
            self,
            input: Optional[Union[Tensor, IntQuantTensor]],
            weight: Optional[Union[Tensor, IntQuantTensor]]) -> Optional[Tensor]:
        if not self.requires_input_scale and self.is_quant_enabled:
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
        if self.is_quant_enabled:
            input_scale = self.compute_bias_scale(input, weight)
            impl = self.export_handler if self.export_mode else self.tensor_quant
            if self.requires_input_scale and input_scale is None and self.is_quant_enabled:
                input_scale = self.scale()
                if input_scale is None:
                    raise RuntimeError("Input scale required")
            elif self.requires_input_scale and input_scale is not None and self.is_quant_enabled:
                input_scale = input_scale.view(-1)

            if self.requires_input_scale and self.is_quant_enabled:
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale)
            else:
                out, out_scale, out_zp, out_bit_width = impl(x)
            if not self.skip_create_quant_tensor:
                out = IntQuantTensor(
                    out, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
                if not self.training and self.cache_inference_quant_bias:
                    cached_bias = _CachedIO(
                        out.detach(), metadata_only=self.cache_inference_quant_bias_metadata_only)
                    self._cached_bias = cached_bias
        else:
            out = x
        return out
