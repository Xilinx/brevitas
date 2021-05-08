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

from warnings import warn
from abc import ABCMeta, abstractmethod
from typing import Type, Union, Optional

from torch.nn import Module
from brevitas.inject import ExtendedInjector, Injector
from brevitas.quant import NoneActQuant
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyProtocol

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
    def is_quant_input_narrow_range(self): # TODO make abstract once narrow range can be cached
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
            proxy_from_injector_impl=ActQuantProxyFromInjector,
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
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            act_impl=act_impl,
            passthrough_act=passthrough_act,
            quant=act_quant,
            proxy_from_injector_impl=ActQuantProxyFromInjector,
            proxy_protocol=ActQuantProxyProtocol,
            none_quant_injector=NoneActQuant,
            proxy_prefix='act_',
            kwargs_prefix='',
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


