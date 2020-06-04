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


SCALING_MIN_VAL = 2.0 ** (-16)


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, compute_output_scale, compute_output_bit_width, return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor
        self._export_mode = False
        # these will be assigned during a normal forward pass; make sure to call
        # .forward with an appropriately-sized input at least once before export
        self.export_in_shape = None
        self.export_out_shape = None
        # these will be assinged during a normal forward pass during inference
        self.export_in_scale = None
        self.export_in_bit_width = None
        self.export_out_scale = None
        self.export_out_bit_width = None

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        self._export_mode = value

    def unpack_input(self, input):
        if self._export_mode:
            if isinstance(input, QuantTensor):
                raise Exception("QuantTensor I/O may not be used during export.")
            else:
                # return cached scale and bit width
                # if input was never QuantTensor, those will be None
                # if input was QuantTensor but now isn't (e.g. due to switch
                # to export mode where all i/o is single tensors), return the
                # cached values
                return input, self.export_in_scale, self.export_in_bit_width
        else:
            if isinstance(input, QuantTensor):
                self.export_in_shape = input.tensor.shape
                # TODO control caching with own config variable
                self.export_in_scale = input[1]
                self.export_in_bit_width = input[2]
                return input
            else:
                self.export_in_shape = input.shape
                return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self._export_mode:
            # do not ever return QuantTensor while exporting
            # cached scale factors will be used in the next layer
            return output
        else:
            self.export_out_shape = output.shape
            # TODO control caching with own config variable
            self.export_out_scale = output_scale
            self.export_out_bit_width = output_bit_width
            if self.return_quant_tensor:
                return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
            else:
                return output
