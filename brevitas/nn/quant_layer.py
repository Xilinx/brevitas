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
from brevitas.quant_tensor import QuantTensor
import brevitas.config as config
import torch

SCALING_MIN_VAL = 2.0 ** (-16)


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, compute_output_scale, compute_output_bit_width, return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor or (config.USE_DYNAMIC_QUANTIZATION and not self.training):
            return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output


class QuantWeightLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, compute_output_scale, compute_output_bit_width, return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor or (config.USE_DYNAMIC_QUANTIZATION and not self.training):
            return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output

    def dynamic_quant(self, weight, input, weight_int, weight_quant, enable_dynamic_quant=False):
        input, input_scale, input_bit_width = self.unpack_input(input)

        if enable_dynamic_quant:
            if weight_int is None:
                weight_int = weight_quant(weight)

            quant_weight, quant_weight_scale, quant_weight_bit_width = weight_int
            if input_scale is not None:
                input = input/input_scale  # Cast to integers
            else:
                quant_weight = quant_weight * quant_weight_scale
        else:
            quant_weight, quant_weight_scale, quant_weight_bit_width = weight_quant(weight)

        quant_weight = (quant_weight, quant_weight_scale, quant_weight_bit_width)
        input = QuantTensor(tensor=input, scale=input_scale, bit_width=input_bit_width)

        return quant_weight, input, weight_int
