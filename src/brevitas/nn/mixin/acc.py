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

from abc import ABCMeta
from typing import Type, Union, Callable, Optional

from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType

from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector, ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol


class QuantAccMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            acc_quant: Union[AccQuantProxyProtocol, Type[Injector]],
            proxy_from_injector_impl:
            Optional[Union[Type[ClampQuantProxyFromInjector], Type[TruncQuantProxyFromInjector]]],
            update_injector: Callable,
            proxy_prefix: str,
            kwargs_prefix: str,
            none_inject: dict,
            **kwargs):

        def update_aqi(aqi):
            if update_injector is not None:
                # don't pass prefix here for retrocompatibility
                return update_injector(self, aqi, kwargs_prefix, **kwargs)
            else:
                return aqi

        proxy_name = proxy_prefix + 'quant'
        if acc_quant is None:
            assert proxy_from_injector_impl is not None
            acc_quant_injector = Injector.let(**none_inject)
            acc_quant_injector = update_aqi(acc_quant_injector)
            acc_quant = proxy_from_injector_impl(acc_quant_injector)
        elif isinstance(acc_quant, type) and issubclass(acc_quant, Injector):
            assert proxy_from_injector_impl is not None
            acc_quant_injector = acc_quant
            acc_quant_injector = update_aqi(acc_quant_injector)
            acc_quant = proxy_from_injector_impl(acc_quant_injector)
        else:
            assert isinstance(acc_quant, AccQuantProxyProtocol)
        setattr(self, proxy_name, acc_quant)


class QuantTruncMixin(QuantAccMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            trunc_quant: Union[AccQuantProxyProtocol, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        super().__init__(
            acc_quant=trunc_quant,
            update_injector=update_injector,
            proxy_from_injector_impl=TruncQuantProxyFromInjector,
            kwargs_prefix='',
            proxy_prefix='trunc_',
            none_inject={'quant_type': QuantType.FP},
            **kwargs)

    @property
    def is_trunc_quant_enabled(self):
        return self.trunc_quant.is_quant_enabled


class QuantClampMixin(QuantAccMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            clamp_quant: Union[AccQuantProxyProtocol, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        super().__init__(
            acc_quant=clamp_quant,
            update_injector=update_injector,
            proxy_from_injector_impl=ClampQuantProxyFromInjector,
            kwargs_prefix='',
            proxy_prefix='clamp_',
            none_inject={'quant_type': QuantType.FP},
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