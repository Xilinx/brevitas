from typing import Tuple

import torch
from torch import Tensor

from brevitas.nn import QuantReLU, QuantHardTanh, QuantIdentity
from .base import FINNQuantInputHandler
from ..function.act import QuantReLUFn, QuantHardTanhFn
from ..utils import finn_datatype


class FINNQuantReLUHandler(FINNQuantInputHandler):
    handled_layer = QuantReLU

    @staticmethod
    def quant_type(module: QuantReLU):
        bit_width = module.quant_act_bit_width()
        signed = module.is_quant_act_signed
        return finn_datatype(bit_width, signed)

    @staticmethod
    def thresholds(module: QuantReLU, extend_tensor_to_channels=True):
        num_distinct_values = 2 ** int(module.quant_act_bit_width().item())
        num_thresholds = num_distinct_values - 1
        flat_scale = module.quant_act_scale().view(-1)
        num_scale_channels = flat_scale.shape[0]
        step = torch.abs(flat_scale)
        min_threshold = step / 2
        thresholds = torch.empty(num_scale_channels, num_thresholds)
        for c in range(num_scale_channels):
            for t in range(num_thresholds):
                thresholds[c][t] = min_threshold[c] + step[c] * t
        if extend_tensor_to_channels:
            output_channels = module._cached_inp.shape[1]
            final_shape = (output_channels, num_thresholds)
            if thresholds.shape != final_shape:
                thresholds = thresholds.expand(final_shape)
        return thresholds

    @staticmethod
    def quant_act_scale(module: QuantReLU):
        quant_act_scale = module.quant_act_scale().type(torch.FloatTensor).detach()
        return quant_act_scale

    def prepare_for_export(self, module: QuantReLU):
        self.symbolic_kwargs = {
            'qnt_type': self.quant_type(module),
            'thres': self.thresholds(module),
            'bias': None,
            'scale': self.quant_act_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantReLUFn.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantHardTanhHandler(FINNQuantInputHandler):
    handled_layer = QuantHardTanh

    @staticmethod
    def quant_type(module: QuantHardTanh):
        return finn_datatype(module.quant_act_bit_width(), module.is_quant_act_signed)

    @staticmethod
    def quant_act_bias(module: QuantHardTanh):
        bit_width = int(module.quant_act_bit_width().item())
        if bit_width == 1:
            return torch.tensor(-0.5).type(torch.FloatTensor)
        else:
            if module.is_quant_act_narrow_range:
                min_non_scaled_val = - (2 ** (bit_width - 1) - 1)
            else:
                min_non_scaled_val = - 2 ** (bit_width - 1)
            return torch.tensor(min_non_scaled_val).type(torch.FloatTensor)

    @staticmethod
    def thresholds(module: QuantHardTanh, extend_tensor_to_channels=True):
        bit_width = int(module.quant_act_bit_width().item())
        if bit_width != 1:
            if module.is_quant_act_narrow_range:
                # assuming narrow range, symmetric quantization around zero
                # when using narrow range, we represent one element less
                num_distinct_values = 2 ** bit_width - 1
            else:
                num_distinct_values = 2 ** bit_width
            num_thresholds = num_distinct_values - 1
            flat_scale = module.quant_act_scale().view(-1)
            num_scale_channels = flat_scale.shape[0]
            step = torch.abs(flat_scale)
            half_step = step / 2.0
            thresholds = torch.empty(num_scale_channels, num_thresholds)
            # compute the value of the smallest threshold, we'll neg-bias all
            # generated thresholds by this much
            min_threshold = - half_step - step * ((num_thresholds // 2) - 1)
            if not module.is_quant_act_narrow_range:
                min_threshold -= step
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    thresholds[c][t] = min_threshold[c] + step[c] * t
            if extend_tensor_to_channels:
                output_channels = module._cached_inp.shape[1]
                final_shape = (output_channels, num_thresholds)
                if thresholds.shape != final_shape:
                    thresholds = thresholds.expand(final_shape)
            return thresholds
        else:
            thresholds = torch.empty([1, 1])
            thresholds[0] = 0
            return thresholds

    @staticmethod
    def quant_act_scale(module: QuantHardTanh):
        bit_width = int(module.quant_act_bit_width().item())
        quant_act_scale = module.quant_act_scale().type(torch.FloatTensor).detach()
        if bit_width != 1:
            return quant_act_scale
        else:
            assert quant_act_scale.view(-1).shape[0] == 1, "Unsupported BIPOLAR per channel scale"
            assert quant_act_scale.flatten().item() == 1.0, "Unsupported BIPOLAR scale != 1"
            return quant_act_scale * 2

    def prepare_for_export(self, module: QuantHardTanh):
        self.symbolic_kwargs = {
            'qnt_type': self.quant_type(module),
            'thres': self.thresholds(module),
            'bias': self.quant_act_bias(module),
            'scale': self.quant_act_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantHardTanhFn.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantIdentityHandler(FINNQuantHardTanhHandler):
    handled_layer = QuantIdentity

