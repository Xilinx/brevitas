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

from typing import Union

from torch import Tensor
from torch.nn import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d
from torch.nn.functional import interpolate

from brevitas.function import round_ste
from brevitas.quant_tensor import QuantTensor
from .mixin.base import QuantLayerMixin


class QuantUpsample(QuantLayerMixin, Upsample):

    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners = None,
            return_quant_tensor: bool = True):
        Upsample.__init__(
            self,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        if self.mode != 'nearest':
            # round interpolated values to scale
            assert x.scale is not None, 'Input scale factor required to interpolate correctly'
            y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingBilinear2d(QuantLayerMixin, UpsamplingBilinear2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingBilinear2d.__init__(
            self,
            size=size,
            scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        # round interpolated values to scale
        assert x.scale is not None, 'Input scale factor required to interpolate correctly'
        y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingNearest2d.__init__(
            self,
            size=size,
            scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        y = x.set(value=y_value)
        return self.pack_output(y)


