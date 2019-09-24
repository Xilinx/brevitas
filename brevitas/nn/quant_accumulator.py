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

from torch.nn import Module

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.proxy.runtime_quant import ClampQuantProxy, TruncQuantProxy
from .quant_layer import QuantLayer


class QuantAccumulator(QuantLayer, Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        QuantLayer.__init__(self,
                            compute_output_scale=True,
                            compute_output_bit_width=True,
                            return_quant_tensor=True)
        Module.__init__(self)

    @property
    def acc_quant_proxy(self):
        return self._act_quant_proxy

    @acc_quant_proxy.setter
    def acc_quant_proxy(self, act_quant_proxy):
        self._acc_quant_proxy = act_quant_proxy

    def forward(self, input):
        tensor, input_scale, input_bit_width = self.unpack_input(input)
        output, output_scale, output_bit_width = self.acc_quant_proxy(tensor, input_scale, input_bit_width)
        return self.pack_output(output, output_scale, output_bit_width)


class ClampQuantAccumulator(QuantAccumulator):

    def __init__(self,
                 ms_bit_width_to_clamp: int = 0,
                 signed: bool = True,
                 narrow_range: bool = True,
                 min_overall_bit_width: int = 2,
                 max_overall_bit_width: int = 32,
                 quant_type: QuantType = QuantType.INT,
                 msb_clamp_bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 per_elem_ops: int = None,
                 clamp_at_least_init_val=False,
                 override_pretrained_bit_width: bool = False):
        super(ClampQuantAccumulator, self).__init__()
        self.per_elem_ops = per_elem_ops
        self.acc_quant_proxy = ClampQuantProxy(signed=signed,
                                               narrow_range=narrow_range,
                                               quant_type=quant_type,
                                               ms_bit_width_to_clamp=ms_bit_width_to_clamp,
                                               min_overall_bit_width=min_overall_bit_width,
                                               max_overall_bit_width=max_overall_bit_width,
                                               msb_clamp_bit_width_impl_type=msb_clamp_bit_width_impl_type,
                                               clamp_at_least_init_val=clamp_at_least_init_val,
                                               override_pretrained_bit_width=override_pretrained_bit_width)


class TruncQuantAccumulator(QuantAccumulator):

    def __init__(self,
                 ls_bit_width_to_trunc: int = 0,
                 signed: bool = True,
                 min_overall_bit_width: int = 2,
                 max_overall_bit_width: int = 32,
                 quant_type: QuantType = QuantType.INT,
                 lsb_trunc_bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 trunc_at_least_init_val=False,
                 explicit_rescaling=False,
                 override_pretrained_bit_width: bool = False):
        super(TruncQuantAccumulator, self).__init__()
        self.acc_quant_proxy = TruncQuantProxy(signed=signed,
                                               quant_type=quant_type,
                                               ls_bit_width_to_trunc=ls_bit_width_to_trunc,
                                               min_overall_bit_width=min_overall_bit_width,
                                               max_overall_bit_width=max_overall_bit_width,
                                               trunc_at_least_init_val=trunc_at_least_init_val,
                                               lsb_trunc_bit_width_impl_type=lsb_trunc_bit_width_impl_type,
                                               override_pretrained_bit_width=override_pretrained_bit_width,
                                               explicit_rescaling=explicit_rescaling)