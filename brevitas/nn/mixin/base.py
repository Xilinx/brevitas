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

from abc import ABCMeta, abstractmethod
from typing import Optional, Type, Union, Callable, Tuple

import torch
from torch.nn import Module
from dependencies import Injector

from brevitas.proxy.parameter_quant import ParameterQuantProxyFromInjector
from brevitas.proxy.parameter_quant import ParameterQuantProxyProtocol
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector, ActQuantProxyProtocol
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector, TruncQuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor


class QuantLayerMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            return_quant_tensor: bool,
            export_mode: bool = False,
            export_handler: Optional = None,
            cache_inference_input_output: bool = False):
        self.return_quant_tensor = return_quant_tensor
        self.export_handler = export_handler
        self.cache_inference_input_output = cache_inference_input_output
        self._export_mode = export_mode
        self._cached_inp_quant_tensor = None
        self._cached_out_quant_tensor = None

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    def export_mode(self):
        if self._export_mode and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        if value:
            self.export_handler.prepare_for_symbolic_execution(self)
        self._export_mode = value

    def unpack_input(self, inp):
        if isinstance(inp, QuantTensor):
            if self.export_mode:
                raise RuntimeError("QuantTensor I/O can't be used during export.")
            if not self.training and self.cache_inference_input_output:
                self._cached_inp_quant_tensor = inp.detach()
            return inp
        else:
            inp = QuantTensor(inp)
            if not self.training and self.cache_inference_input_output:
                self._cached_inp_quant_tensor = inp.detach()
            return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_input_output:
            self._cached_out_quant_tensor = quant_output.detach()
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class QuantParameterMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            parameter: torch.nn.Parameter,
            parameter_quant: Optional[Union[ParameterQuantProxyProtocol, Type[Injector]]],
            proxy_from_injector_impl: Optional[Type[ParameterQuantProxyFromInjector]],
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
            assert proxy_from_injector_impl is not None
            parameter_quant_injector = Injector.let(tensor_quant=None)
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_from_injector_impl(parameter_quant_injector)
        elif issubclass(parameter_quant, Injector):
            assert proxy_from_injector_impl is not None
            parameter_quant_injector = parameter_quant
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_from_injector_impl(parameter_quant_injector)
        else:
            assert isinstance(parameter_quant, ParameterQuantProxyProtocol)
        setattr(self, proxy_name, parameter_quant)
        getattr(self, proxy_name).add_tracked_parameter(parameter)


class QuantActMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Optional[Module],
            act_quant: Union[ActQuantProxyProtocol, Type[Injector]],
            proxy_from_injector_impl: Optional[Type[ActQuantProxyFromInjector]],
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
            act_quant = proxy_from_injector_impl(act_quant_injector)
        elif issubclass(act_quant, Injector):
            assert proxy_from_injector_impl is not None
            act_quant_injector = act_quant
            if 'act_impl' not in act_quant_injector or act_quant_injector.act_impl is None:
                act_quant_injector = act_quant_injector.let(act_impl=act_impl)
            act_quant_injector = update_aqi(act_quant_injector)
            act_quant = proxy_from_injector_impl(act_quant_injector)
        else:
            assert isinstance(act_quant, ActQuantProxyProtocol)
        setattr(self, proxy_name, act_quant)


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
        elif issubclass(acc_quant, Injector):
            assert proxy_from_injector_impl is not None
            acc_quant_injector = acc_quant
            acc_quant_injector = update_aqi(acc_quant_injector)
            acc_quant = proxy_from_injector_impl(acc_quant_injector)
        else:
            assert isinstance(acc_quant, AccQuantProxyProtocol)
        setattr(self, proxy_name, acc_quant)