# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas.function import max_int
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector
from .quant_proxy import QuantProxyProtocol

__all__ = [
    'WeightQuantProxyFromInjector',
    'BiasQuantProxyFromInjector',
    'WeightQuantProxyProtocol',
    'BiasQuantProxyProtocol']


@runtime_checkable
class WeightQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        ...


@runtime_checkable
class BiasQuantProxyProtocol(QuantProxyProtocol, Protocol):
    requires_input_bit_width: bool
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

    def init_tensor_quant(self):
        param_list = self.tracked_parameter_list
        # params might not be there yet, e.g. bias before merging
        if param_list:
            self.quant_injector = self.quant_injector.let(tracked_parameter_list=param_list)
            super(ParameterQuantProxyFromInjector, self).init_tensor_quant()

    def max_uint_value(self, bit_width):
        return max_int(False, self.is_narrow_range, bit_width)


class WeightQuantProxyFromInjector(ParameterQuantProxyFromInjector, WeightQuantProxyProtocol):

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    @property
    def requires_quant_input(self):
        return False

    def scale(self):
        scale = self.__call__(self.tracked_parameter_list[0]).scale
        return scale

    def zero_point(self):
        zero_point = self.__call__(self.tracked_parameter_list[0]).zero_point
        return zero_point

    def bit_width(self):
        bit_width_ = self.__call__(self.tracked_parameter_list[0]).bit_width
        return bit_width_

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width = impl(x)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)


class DecoupledWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    def pre_scale(self):
        output_tuple = self.tensor_quant(self.tracked_parameter_list[0])
        out, scale, zero_point, bit_width, pre_scale, pre_zero_point = output_tuple
        return pre_scale

    def pre_zero_point(self):
        output_tuple = self.tensor_quant(self.tracked_parameter_list[0])
        out, scale, zero_point, bit_width, pre_scale, pre_zero_point = output_tuple
        return pre_zero_point

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width, pre_scale, pre_zero_point = impl(x)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)


class DecoupledWeightQuantWithInputProxyFromInjector(DecoupledWeightQuantProxyFromInjector):

    @property
    def requires_quant_input(self):
        return True

    def scale(self):
        raise NotImplementedError

    def zero_point(self):
        raise NotImplementedError

    def bit_width(self):
        raise NotImplementedError

    def pre_scale(self):
        raise NotImplementedError

    def pre_zero_point(self):
        raise NotImplementedError

    def forward(
            self, x: torch.Tensor, input_bit_width: torch.Tensor,
            input_is_signed: bool) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width, pre_scale, pre_zero_point = impl(x, input_bit_width, input_is_signed)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)


class BiasQuantProxyFromInjector(ParameterQuantProxyFromInjector, BiasQuantProxyProtocol):

    @property
    def tracked_parameter_list(self):
        return [m.bias for m in self.tracked_module_list if m.bias is not None]

    @property
    def requires_input_bit_width(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_bit_width
        else:
            return False

    @property
    def requires_input_scale(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_scale
        else:
            return False

    def scale(self):
        if self.requires_input_scale:
            return None
        zhs = self._zero_hw_sentinel()
        scale = self.__call__(self.tracked_parameter_list[0], zhs, zhs).scale
        return scale

    def zero_point(self):
        zhs = self._zero_hw_sentinel()
        zero_point = self.__call__(self.tracked_parameter_list[0], zhs, zhs).zero_point
        return zero_point

    def bit_width(self):
        if self.requires_input_bit_width:
            return None
        zhs = self._zero_hw_sentinel()
        bit_width = self.__call__(self.tracked_parameter_list[0], zhs, zhs).bit_width
        return bit_width

    def forward(
            self,
            x: Tensor,
            input_scale: Optional[Tensor] = None,
            input_bit_width: Optional[Tensor] = None) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            if self.requires_input_scale and input_scale is None:
                raise RuntimeError("Input scale required")
            if self.requires_input_bit_width and input_bit_width is None:
                raise RuntimeError("Input bit-width required")
            if self.requires_input_scale and self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale, input_bit_width)
            elif self.requires_input_scale and not self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale)
            elif not self.requires_input_scale and not self.requires_input_bit_width:
                out, out_scale, out_zp, out_bit_width = impl(x)
            else:
                raise RuntimeError("Internally defined bit-width required")
            return QuantTensor(out, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        else:
            return QuantTensor(x, training=self.training)
