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
from typing import Type, Union, Optional

from brevitas.inject import Injector, ExtendedInjector
from brevitas.quant import NoneClampQuant, NoneTruncQuant
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector, ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from .base import QuantProxyMixin


AccQuantType = Union[AccQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class QuantTruncMixin(QuantProxyMixin):
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
            proxy_from_injector_impl=ClampQuantProxyFromInjector,
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