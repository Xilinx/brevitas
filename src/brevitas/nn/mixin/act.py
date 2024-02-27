# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Type, Union

from torch.nn import Module

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
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
