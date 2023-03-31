# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Type, Union
from warnings import warn

from torch.nn import Module

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyProtocol
from brevitas.quant import NoneActQuant

from .base import QuantProxyMixin

ActQuantType = Union[ActQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class QuantInputMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, input_quant: Optional[ActQuantType], **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=input_quant,
            proxy_protocol=ActQuantProxyProtocol,
            none_quant_injector=NoneActQuant,
            proxy_prefix='input_',
            kwargs_prefix='input_',
            input_act_impl=None,
            input_passthrough_act=True,
            **kwargs)

    @property
    def is_input_quant_enabled(self):
        return self.input_quant.is_quant_enabled

    @property
    def is_quant_input_narrow_range(self):  # TODO make abstract once narrow range can be cached
        return self.input_quant.is_narrow_range

    @property
    @abstractmethod
    def is_quant_input_signed(self):
        pass

    @abstractmethod
    def quant_input_scale(self):
        pass

    @abstractmethod
    def quant_input_zero_point(self):
        pass

    @abstractmethod
    def quant_input_bit_width(self):
        pass


class QuantOutputMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, output_quant: Optional[ActQuantType], **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=output_quant,
            proxy_protocol=ActQuantProxyProtocol,
            none_quant_injector=NoneActQuant,
            proxy_prefix='output_',
            kwargs_prefix='output_',
            output_act_impl=None,
            output_passthrough_act=True,
            **kwargs)

    @property
    def is_output_quant_enabled(self):
        return self.output_quant.is_quant_enabled

    @property
    def is_quant_output_narrow_range(self):  # TODO make abstract once narrow range can be cached
        return self.output_quant.is_narrow_range

    @property
    @abstractmethod
    def is_quant_output_signed(self):
        pass

    @abstractmethod
    def quant_output_scale(self):
        pass

    @abstractmethod
    def quant_output_zero_point(self):
        pass

    @abstractmethod
    def quant_output_bit_width(self):
        pass


class QuantNonLinearActMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Optional[Type[Module]],
            passthrough_act: bool,
            act_quant: Optional[ActQuantType],
            act_proxy_prefix='act_',
            act_kwargs_prefix='',
            **kwargs):
        prefixed_kwargs = {
            act_kwargs_prefix + 'act_impl': act_impl,
            act_kwargs_prefix + 'passthrough_act': passthrough_act}
        QuantProxyMixin.__init__(
            self,
            quant=act_quant,
            proxy_prefix=act_proxy_prefix,
            kwargs_prefix=act_kwargs_prefix,
            proxy_protocol=ActQuantProxyProtocol,
            none_quant_injector=NoneActQuant,
            **prefixed_kwargs,
            **kwargs)

    @property
    def is_act_quant_enabled(self):
        return self.act_quant.is_quant_enabled

    @property
    def is_quant_act_narrow_range(self):  # TODO make abstract once narrow range can be cached
        return self.act_quant.is_narrow_range

    @property
    @abstractmethod
    def is_quant_act_signed(self):
        pass

    @abstractmethod
    def quant_act_scale(self):
        pass

    @abstractmethod
    def quant_act_zero_point(self):
        pass

    @abstractmethod
    def quant_act_bit_width(self):
        pass
