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
from typing import Optional, Type, Union, Callable

import torch
from torch.nn import Module
from dependencies import Injector

from brevitas.proxy.parameter_quant import ParameterQuantProxy
from brevitas.proxy.runtime_quant import ActQuantProxy, TruncQuantProxy, ClampQuantProxy
from brevitas.quant_tensor import QuantTensor


class QuantLayerMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, inp):
        if isinstance(inp, QuantTensor):
            return inp
        else:
            return QuantTensor(inp)

    def pack_output(self, quant_output: QuantTensor):
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class QuantParameterMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            parameter: torch.nn.Parameter,
            parameter_quant: Optional[Union[ParameterQuantProxy, Type[Injector]]],
            proxy_impl: Optional[Type[ParameterQuantProxy]],
            update_injector: Optional[Callable],
            prefix: str,
            **kwargs):

        def update_pqi(pqi):
            if update_injector is not None:
                return update_injector(self, pqi, prefix, **kwargs)
            else:
                return pqi

        proxy_name = prefix + 'quant'
        if parameter_quant is None:
            assert proxy_impl is not None
            parameter_quant_injector = Injector.let(tensor_quant=None)
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_impl(parameter_quant_injector)
        elif isinstance(parameter_quant, ParameterQuantProxy):
            pass
        else:
            assert proxy_impl is not None
            parameter_quant_injector = parameter_quant
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_impl(parameter_quant_injector)
        setattr(self, proxy_name, parameter_quant)
        getattr(self, proxy_name).add_tracked_parameter(parameter)


class QuantActMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Optional[Module],
            act_quant: Union[ActQuantProxy, Type[Injector]],
            proxy_impl: Optional[Type[ActQuantProxy]],
            update_injector: Callable,
            proxy_prefix: str,
            kwargs_prefix: str,
            **kwargs):

        def update_aqi(aqi):
            if update_injector is not None:
                # don't pass prefix here for retrocompatibility
                return update_injector(self, aqi, kwargs_prefix, **kwargs)
            else:
                return aqi

        proxy_name = proxy_prefix + 'quant'
        if act_quant is None:
            act_quant_injector = Injector.let(tensor_quant=None)
            act_quant_injector = act_quant_injector.let(act_impl=act_impl)
            act_quant_injector = update_aqi(act_quant_injector)
            act_quant = proxy_impl(act_quant_injector)
        elif isinstance(act_quant, proxy_impl):
            pass
        else:
            assert proxy_impl is not None
            act_quant_injector = act_quant
            act_quant_injector = update_aqi(act_quant_injector)
            act_quant = proxy_impl(act_quant_injector)
        setattr(self, proxy_name, act_quant)


class QuantAccMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            acc_quant: Union[TruncQuantProxy, ClampQuantProxy, Type[Injector]],
            proxy_impl: Optional[Union[Type[TruncQuantProxy], Type[ClampQuantProxy]]],
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
            assert proxy_impl is not None
            acc_quant_injector = Injector.let(none_inject)
            acc_quant_injector = update_aqi(acc_quant_injector)
            acc_quant = proxy_impl(acc_quant_injector)
        elif isinstance(acc_quant, proxy_impl):
            pass
        else:
            assert proxy_impl is not None
            acc_quant_injector = acc_quant
            acc_quant_injector = update_aqi(acc_quant_injector)
            acc_quant = proxy_impl(acc_quant_injector)
        setattr(self, proxy_name, acc_quant)