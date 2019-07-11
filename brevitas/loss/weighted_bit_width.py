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

from typing import List
from abc import ABCMeta, abstractmethod

from functools import reduce
from operator import mul

import torch
from torch import nn

from brevitas.utils.quant_utils import *
from brevitas.nn.quant_linear import QuantLinear
from brevitas.nn.quant_conv import QuantConv2d
from brevitas.nn.quant_avg_pool import QuantAvgPool2d

MEGA = 10e6

class BitWidthWeighted(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model: nn.Module = model
        self.weighted_bit_width_list: List[torch.Tensor] = []
        self.tot_num_elements: int = 0
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def zero_accumulated_values(self):
        del self.weighted_bit_width_list
        del self.tot_num_elements
        self.weighted_bit_width_list = []
        self.tot_num_elements = 0

    def retrieve(self, as_average=True):
        if self.tot_num_elements != 0 and self.weighted_bit_width_list:
            if as_average:
                value = sum(self.weighted_bit_width_list) / self.tot_num_elements
            else:
                value = [bit_width / self.tot_num_elements for bit_width in self.weighted_bit_width_list]
        else:
            raise Exception("Number of elements to penalize can't be zero")
        return value

    def log(self):
        return self.retrieve(as_average=True).detach().clone()


class WeightBitWidthWeightedBySize(BitWidthWeighted):

    def __init__(self, model):
        super(WeightBitWidthWeightedBySize, self).__init__(model=model)
        pass

    def register_hooks(self):

        def hook_fn(module, input, output):
            (quant_weight, output_scale, output_bit_width) = output
            num_elements = reduce(mul, quant_weight.size(), 1)
            self.weighted_bit_width_list.append(num_elements * output_bit_width)
            self.tot_num_elements += num_elements

        for name, module in self.model.named_modules():
            if has_learned_weight_bit_width(module):
                module.register_forward_hook(hook_fn)


class ActivationBitWidthWeightedBySize(BitWidthWeighted):

    def __init__(self, model):
        super(ActivationBitWidthWeightedBySize, self).__init__(model=model)
        pass

    def register_hooks(self):

        def hook_fn(module, input, output):
            (quant_act, output_scale, output_bit_width) = output
            num_elements = reduce(mul, quant_act.size()[1:], 1)  # exclude batch size
            self.weighted_bit_width_list.append(num_elements * output_bit_width)
            self.tot_num_elements += num_elements

        for name, module in self.model.named_modules():
            if has_learned_activation_bit_width(module):
                module.register_forward_hook(hook_fn)


class QuantLayerOutputBitWidthWeightedByOps(BitWidthWeighted):

    def __init__(self, model, layer_types=(QuantConv2d, QuantLinear)):
        self.layer_types = layer_types
        super(QuantLayerOutputBitWidthWeightedByOps, self).__init__(model=model)
        pass

    def register_hooks(self):

        def hook_fn(module, input, output):
            if module.return_quant_tensor:
                output, output_scale, output_bit_width = output
            output_size = reduce(mul, output.size()[1:], 1)  # exclude batch size
            num_mops = output_size * module.per_elem_ops / MEGA
            self.weighted_bit_width_list.append(output_bit_width * num_mops)
            self.tot_num_elements += num_mops

        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types) \
                    and module.return_quant_tensor \
                    and module.compute_output_bit_width\
                    and module.per_elem_ops is not None:
                module.register_forward_hook(hook_fn)



