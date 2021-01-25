from typing import List
from abc import ABCMeta, abstractmethod

from functools import reduce
from operator import mul

import torch
from torch import nn

from brevitas.utils.quant_utils import *
from brevitas.nn.quant_linear import QuantLinear
from brevitas.nn.quant_conv import QuantConv2d
from brevitas.quant_tensor import QuantTensor

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

        def hook_fn(module, input, output: QuantTensor):
            num_elements = reduce(mul, output.value.size(), 1)
            self.weighted_bit_width_list.append(num_elements * output.bit_width)
            self.tot_num_elements += num_elements

        for name, module in self.model.named_modules():
            if has_learned_weight_bit_width(module):
                module.register_forward_hook(hook_fn)


class ActivationBitWidthWeightedBySize(BitWidthWeighted):

    def __init__(self, model):
        super(ActivationBitWidthWeightedBySize, self).__init__(model=model)
        pass

    def register_hooks(self):

        def hook_fn(module, input, output: QuantTensor):
            num_elements = reduce(mul, output.value.size()[1:], 1)  # exclude batch size
            self.weighted_bit_width_list.append(num_elements * output.bit_width)
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
            if isinstance(output, QuantTensor):
                output_size = reduce(mul, output.size()[1:], 1)  # exclude batch size
                num_mops = output_size * module.per_elem_ops / MEGA
                self.weighted_bit_width_list.append(output.bit_width * num_mops)
                self.tot_num_elements += num_mops

        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types) \
                    and module.return_quant_tensor \
                    and module.per_elem_ops is not None:
                module.register_forward_hook(hook_fn)



