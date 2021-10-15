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
import torch.jit
from torch import Tensor

from warnings import warn
from abc import ABCMeta, abstractmethod
from typing import Optional, Union
from inspect import isclass

from brevitas import config
from brevitas.inject import ExtendedInjector, Injector
from brevitas.quant_tensor import QuantTensor

from .utils import filter_kwargs


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


class QuantProxyMixin(object):
    __metaclass__ = ABCMeta


    def __init__(
            self,
            quant,
            proxy_protocol,
            none_quant_injector,
            proxy_prefix: str,
            kwargs_prefix: str,
            **kwargs):
        proxy_name = proxy_prefix + 'quant'
        if quant is None:
            quant_injector = none_quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        elif isclass(quant) and issubclass(quant, (Injector, ExtendedInjector)):
            quant_injector = quant
            quant_injector = quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        else:
            if not isinstance(quant, proxy_protocol):
                raise RuntimeError(
                    "The quantizer passed does not adhere to the quantization protocol.")
            quant.add_tracked_module(self)
            if filter_kwargs(kwargs_prefix, kwargs):
                warn('Keyword arguments are being passed but they not being used.')
        setattr(self, proxy_name, quant)


class QuantLayerMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            return_quant_tensor: bool,
            export_mode: bool = False,
            export_debug_name: Optional[str] = None,
            export_handler: Optional = None,
            cache_inference_quant_inp: bool = False,
            cache_inference_quant_out: bool = False,
            cache_quant_io_metadata_only: bool = True):
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor
        self.export_handler = export_handler
        self.cache_inference_quant_inp = cache_inference_quant_inp
        self.cache_inference_quant_out = cache_inference_quant_out
        self.cache_quant_io_metadata_only = cache_quant_io_metadata_only
        self._export_mode = export_mode
        self._export_debug_name = export_debug_name
        self._cached_inp = None
        self._cached_out = None
        self.export_input_debug = False
        self.export_output_debug = False

    @property
    def export_debug_name(self):
        return self._export_debug_name

    @export_debug_name.setter
    def export_debug_name(self, value):
        self._export_debug_name = value

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    @abstractmethod
    def requires_export_handler(self):
        pass

    @property
    def export_mode(self):
        if self._export_mode and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return  # don't set export mode when it's not required and there is no handler
        elif value and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler.reset()
        self._export_mode = value

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    def _set_global_is_quant_layer(self, value):
        config._IS_INSIDE_QUANT_LAYER = value

    def quant_input_scale(self):
        if self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    @property
    def is_quant_output_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_output_scale(self):
        if self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_zero_point(self):
        if self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_bit_width(self):
        if self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]):
        self._set_global_is_quant_layer(True)
        # Hack to recognize a QuantTensor that has decayed to a tuple
        # when used as input to tracing (e.g. during ONNX export)
        if (torch._C._get_tracing_state() is not None
                and isinstance(inp, tuple)
                and len(inp) == len(QuantTensor._fields)
                and all([isinstance(t, Tensor) for t in inp])):
            inp = QuantTensor(*inp)
        if isinstance(inp, QuantTensor):
            # don't cache values during export pass
            if not self.training and not self._export_mode and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp
        else:
            inp = QuantTensor(inp, training=self.training)
            if not self.training and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_quant_out:
            self._cached_out = _CachedIO(quant_output.detach(), self.cache_quant_io_metadata_only)
        self._set_global_is_quant_layer(False)
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value
