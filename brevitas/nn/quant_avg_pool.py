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

from typing import Optional

import math
import torch
from torch.nn import AvgPool2d

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.function.ops import ceil_ste, max_uint
from brevitas.nn.quant_layer import QuantLayer
from brevitas.proxy.runtime_quant import TruncQuantProxy
from brevitas.quant_tensor import pack_quant_tensor


class QuantAvgPool2d(QuantLayer, AvgPool2d):

    def __init__(self,
                 kernel_size: int,
                 stride: int = None,
                 signed: bool = True,
                 min_overall_bit_width: Optional[int] = 2,
                 max_overall_bit_width: Optional[int] = 32,
                 quant_type: QuantType = QuantType.FP,
                 lsb_trunc_bit_width_impl_type = BitWidthImplType.CONST):
        QuantLayer.__init__(self,
                            compute_output_scale=True,
                            compute_output_bit_width=True,
                            return_quant_tensor=True)
        AvgPool2d.__init__(self,
                           kernel_size=kernel_size,
                           stride=stride)
        ls_bit_width_to_trunc = math.ceil(math.log2(kernel_size * kernel_size))
        self.signed = signed
        self.quant_type = quant_type
        explicit_rescaling = True  # we are explicitly rescaling as we are replacing the div in avg with trunc
        self.accumulator_quant = TruncQuantProxy(signed=signed,
                                                 quant_type=quant_type,
                                                 trunc_at_least_init_val=True,
                                                 ls_bit_width_to_trunc=ls_bit_width_to_trunc,
                                                 min_overall_bit_width=min_overall_bit_width,
                                                 max_overall_bit_width=max_overall_bit_width,
                                                 lsb_trunc_bit_width_impl_type=lsb_trunc_bit_width_impl_type,
                                                 explicit_rescaling=explicit_rescaling,
                                                 override_pretrained_bit_width=False)

    def forward(self, input):
        input_tensor, input_scale, input_bit_width = self.unpack_input(input)
        x = super(QuantAvgPool2d, self).forward(input_tensor)
        if self.quant_type != QuantType.FP:
            x = x * (self.kernel_size * self.kernel_size)  # remove scaling introduced by average
            output_bit_width = self.max_output_bit_width(input_bit_width)
            x, output_scale, output_bit_width = self.accumulator_quant(x, input_scale, output_bit_width)
            return pack_quant_tensor(x, output_scale, output_bit_width)
        else:
            return pack_quant_tensor(x, input_scale, input_bit_width)

    def max_output_bit_width(self, input_bit_width):
        max_uint_input = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_uint_output = max_uint_input * self.kernel_size * self.kernel_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
