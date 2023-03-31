# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from typing import Optional, Type, Union

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.quant import NoneClampQuant
from brevitas.quant import NoneTruncQuant

from .base import QuantProxyMixin

AccQuantType = Union[AccQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class TruncMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, trunc_quant: Optional[AccQuantType], **kwargs):
        super().__init__(
            quant=trunc_quant,
            proxy_protocol=AccQuantProxyProtocol,
            none_quant_injector=NoneTruncQuant,
            kwargs_prefix='',
            proxy_prefix='trunc_',
            **kwargs)

    @property
    def is_trunc_quant_enabled(self):
        return self.trunc_quant.is_quant_enabled


class QuantClampMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, clamp_quant: Optional[AccQuantType], **kwargs):
        super().__init__(
            quant=clamp_quant,
            proxy_protocol=AccQuantProxyProtocol,
            none_quant_injector=NoneClampQuant,
            kwargs_prefix='',
            proxy_prefix='clamp_',
            **kwargs)

    @property
    def is_clamp_quant_enabled(self):
        return self.clamp_quant.is_quant_enabled

    @property
    def is_quant_clamp_narrow_range(self):
        assert self.is_clamp_quant_enabled
        return self.clamp_quant.is_narrow_range

    @property
    def is_quant_clamp_signed(self):
        assert self.is_clamp_quant_enabled
        return self.clamp_quant.is_signed
