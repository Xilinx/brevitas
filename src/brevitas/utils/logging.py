# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from functools import partial
from typing import Dict

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
