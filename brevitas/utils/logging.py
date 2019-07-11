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

from typing import Dict
from abc import ABCMeta, abstractmethod
from functools import partial

import torch
from torch import nn

from brevitas.utils.quant_utils import *


class LogBitWidth(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model: nn.Module = model
        self.bit_width_dict: Dict[str, int] = {}
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass


class LogWeightBitWidth(LogBitWidth):

    def __init__(self, model):
        super(LogWeightBitWidth, self).__init__(model=model)
        pass

    def register_hooks(self):
        for name, module in self.model.named_modules():
            def hook_fn(module, input, output, name):
                (quant_weight, scale, bit_width) = output
                self.bit_width_dict[name] = bit_width.detach().clone()
            if has_learned_weight_bit_width(module):
                module.register_forward_hook(partial(hook_fn, name=name))


class LogActivationBitWidth(LogBitWidth):

    def __init__(self, model):
        super(LogActivationBitWidth, self).__init__(model=model)
        pass

    def register_hooks(self):
        for name, module in self.model.named_modules():
            def hook_fn(module, input, output, name):
                (quant_act, scale, bit_width) = output
                self.bit_width_dict[name] = bit_width.detach().clone()
            if has_learned_activation_bit_width(module):
                module.register_forward_hook(partial(hook_fn, name=name))