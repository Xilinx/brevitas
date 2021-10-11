from abc import ABC, abstractmethod

import torch
from torch import Tensor

from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas.nn import QuantIdentity, QuantReLU
from brevitas.nn import QuantConvTranspose2d, QuantConv2d, QuantLinear
from brevitas.export.onnx.handler import Kernel2dApplHandlerMixin
from ..handler import DPUQuantWBIOLHandler, DPUQuantLayerHandler
from .function import XIRFixFn, XIRGemmFn
from .function import XIRConv2dFn, XIRConvTranpose2dFn


class XIRQuantActHandler(DPUQuantLayerHandler, ABC):

    @property
    @abstractmethod
    def act_impl(self):
        pass

    @classmethod
    @abstractmethod
    def act_symbolic_kwargs(cls, module: QuantNLAL):
        pass

    @classmethod
    def act_quant_symbolic_kwargs(cls, module: QuantNLAL):
        if module.is_act_quant_enabled:
            act_quant_kwargs = {
                'bit_width': cls.quant_output_bit_width(module),
                'fix_point': cls.quant_output_scale(module),
                'signed': module.is_quant_output_signed,
            }
        else:
            act_quant_kwargs = None
        return act_quant_kwargs

    def prepare_for_export(self, module: QuantReLU):
            self.symbolic_kwargs = {
                'act_quant': self.act_quant_symbolic_kwargs(module),
                'act': self.act_symbolic_kwargs(module)}

    def symbolic_execution(self, x: Tensor):
        act_kwargs = self.symbolic_kwargs['act']
        act_quant_kwargs = self.symbolic_kwargs['act_quant']
        if self.act_impl is not None:
            x = self.act_impl(x, *act_kwargs.values())
        if act_quant_kwargs is not None:
            x = XIRFixFn.apply(x, *act_quant_kwargs.values())
        return x


class XIRQuantReLUHandler(XIRQuantActHandler):
    handled_layer = QuantReLU

    @property
    def act_impl(self):
        return torch.relu

    def act_symbolic_kwargs(cls, module: QuantReLU):
        return {}


class XIRQuantIdentityHandler(XIRQuantActHandler):
    handled_layer = QuantIdentity

    @property
    def act_impl(self):
        return None

    def act_symbolic_kwargs(cls, module: QuantReLU):
        return {}


class XIRQuantWBIOLHandler(DPUQuantWBIOLHandler, ABC):

    @property
    @abstractmethod
    def op_impl(self):
        pass

    @classmethod
    @abstractmethod
    def op_symbolic_kwargs(cls, module: QuantWBIOL):
        pass

    @classmethod
    def input_quant_symbolic_kwargs(cls, module: QuantWBIOL):
        if module.is_input_quant_enabled:
            input_quant_kwargs = {
                'bit_width': cls.quant_input_bit_width(module),
                'fix_point': cls.quant_input_scale(module),
                'signed': module.is_quant_input_signed,
            }
        else:
            input_quant_kwargs = None
        return input_quant_kwargs

    @classmethod
    def weight_quant_symbolic_kwargs(cls, module: QuantWBIOL):
        if module.is_weight_quant_enabled:
            weight_quant_kwargs = {
                'bit_width': cls.quant_weight_bit_width(module),
                'fix_point': cls.quant_weight_scale(module),
                'signed': module.is_quant_weight_signed,
            }
        else:
            weight_quant_kwargs = None
        return weight_quant_kwargs

    @classmethod
    def bias_quant_symbolic_kwargs(cls, module: QuantWBIOL):
        if module.is_bias_quant_enabled:
            bias_quant_kwargs = {
                'bit_width': cls.quant_bias_bit_width(module),
                'fix_point': cls.quant_bias_scale(module),
                'signed': module.is_quant_bias_signed,
            }
        else:
            bias_quant_kwargs = None
        return bias_quant_kwargs

    @classmethod
    def output_quant_symbolic_kwargs(cls, module: QuantWBIOL):
        if module.is_output_quant_enabled:
            output_quant_kwargs = {
                'bit_width': cls.quant_output_bit_width(module),
                'fix_point': cls.quant_output_scale(module),
                'signed': module.is_quant_output_signed,
            }
        else:
            output_quant_kwargs = None
        return output_quant_kwargs

    def prepare_for_export(self, module: QuantConv2d):
        bias = module.quant_bias()
        if bias is not None:
           bias = bias.value.detach()
        weight = module.quant_weight().value.detach()
        if len(weight.shape) == 4:  # move weights to NHWC already
            weight = weight.permute(0, 2, 3, 1)
        self.symbolic_kwargs = {
            'weight': weight,
            'bias': bias,
            'input_quant': self.input_quant_symbolic_kwargs(module),
            'weight_quant': self.weight_quant_symbolic_kwargs(module),
            'bias_quant': self.bias_quant_symbolic_kwargs(module),
            'output_quant': self.output_quant_symbolic_kwargs(module),
            'op': self.op_symbolic_kwargs(module)}

    def symbolic_execution(self, x):
        weight = self.symbolic_kwargs['weight']
        bias = self.symbolic_kwargs['bias']
        input_quant_kwargs = self.symbolic_kwargs['input_quant']
        weight_quant_kwargs = self.symbolic_kwargs['weight_quant']
        bias_quant_kwargs = self.symbolic_kwargs['bias_quant']
        output_quant_kwargs = self.symbolic_kwargs['output_quant']
        op_kwargs = self.symbolic_kwargs['op']
        if input_quant_kwargs is not None:
           x = XIRFixFn.apply(x, *input_quant_kwargs.values())
        if weight_quant_kwargs is not None:
            weight = XIRFixFn.apply(weight, *weight_quant_kwargs.values())
        if bias is not None and bias_quant_kwargs is not None:
            bias = XIRFixFn.apply(bias, *bias_quant_kwargs.values())
        out = self.op_impl(x, weight, bias, *op_kwargs.values())
        if output_quant_kwargs is not None:
            out = XIRFixFn.apply(out, *output_quant_kwargs.values())
        return out


class XIRQuantConv2dHandler(XIRQuantWBIOLHandler, Kernel2dApplHandlerMixin):
    handled_layer = QuantConv2d

    @property
    def op_impl(self):
        return XIRConv2dFn.apply

    @classmethod
    def op_symbolic_kwargs(cls, module: QuantConv2d):
        op_symbolic_kwargs = {
            'is_depthwise': module.channelwise_separable,
            'kernel_size': cls.kernel_shape(module),
            'padding': cls.padding(module),
            'padding_type': module.padding_type,
            'stride': cls.stride(module),
            'dilation': cls.dilation(module),
            'output_shape': cls.quant_output_shape(module)}
        return op_symbolic_kwargs


class XIRQuantConvTranspose2dHandler(XIRQuantWBIOLHandler):
    handled_layer = QuantConvTranspose2d

    @property
    def op_impl(self):
        return XIRConvTranpose2dFn.apply

    @classmethod
    def op_symbolic_kwargs(cls, module: QuantConvTranspose2d):
        op_symbolic_kwargs = {
            'is_depthwise': module.channelwise_separable,
            'kernel_size': cls.kernel_shape(module),
            'padding': cls.padding(module),
            'stride': cls.stride(module),
            'dilation': cls.dilation(module),
            'output_shape': cls.quant_output_shape(module)}
        return op_symbolic_kwargs


class XIRQuantLinearHandler(XIRQuantWBIOLHandler):
    handled_layer = QuantLinear

    @property
    def op_impl(self):
        return XIRGemmFn.apply

    @classmethod
    def op_symbolic_kwargs(cls, module: QuantLinear):
        return {}