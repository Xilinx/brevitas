# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import StaticScaledQuantizer
import torch
from torch import Tensor
import torch.nn as nn

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
import brevitas.nn as qnn
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.proxy import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class SharkActEqualization(nn.Module):
    handled_layer = EqualizedModule

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if hasattr(module.layer, 'export_handler') and module.layer.export_handler is not None:
            module.layer.export_handler.layer_name = self.layer_name
        self.premul_input = module.scale.weight.contiguous()

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        premul_input = DefaultPrimitiveTensor(
            name=f"{self.layer_name}.premul_input",
            data=self.premul_input,
        )
        self.shared_dict[premul_input.name] = premul_input
        return premul_input


class SharkWeightQuantMixin:

    @staticmethod
    def prepare_weight_for_export(module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            scale = module.scale().contiguous()
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point -
                                                                            128.).to(scale.device)
            zero_point = zero_point
            quant_metadata = {'scale': scale, 'zero_point': zero_point}
            if isinstance(module, WeightQuantProxyFromInjector):
                assert module.bit_width() == 8., "Only Int8 is supported for export"
                quant_metadata['dtype'] = torch.int8
            elif isinstance(module, WeightFloatQuantProxyFromInjector):
                if module.is_ocp_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2
                elif module.is_ocp_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fn
                elif module.is_fnuz_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2fnuz
                elif module.is_fnuz_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fnuz
                else:
                    raise ValueError("Dtype not supported for export")
            return quant_metadata
        else:
            return None

    @staticmethod
    def weight_quant(x, quant_metadata):
        scale = quant_metadata['scale']
        zero_point = quant_metadata['zero_point']
        layer_name = quant_metadata['layer_name']
        shared_dict = quant_metadata['shared_dict']
        dtype = quant_metadata['dtype']

        weight_quant = StaticScaledQuantizer(
            name=layer_name,
            scale=torch.reciprocal(scale),
            reciprocal_scale=scale,
            offset=zero_point,
            dtype=dtype)
        quant_weight = weight_quant.quantize(x, name=layer_name)
        shared_dict[layer_name] = quant_weight
        return x


class SharkActQuantMixin:

    @staticmethod
    def prepare_act_for_export(module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            scale = module.scale().contiguous()
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point -
                                                                            128.).to(scale.device)
            zero_point = zero_point
            quant_metadata = {'scale': scale, 'zero_point': zero_point}
            if isinstance(module, ActQuantProxyFromInjector):
                assert module.bit_width() == 8., "Only Int8 is supported for export"
                quant_metadata['dtype'] = torch.int8
            elif isinstance(module, ActFloatQuantProxyFromInjector):
                if module.is_ocp_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2
                elif module.is_ocp_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fn
                elif module.is_fnuz_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2fnuz
                elif module.is_fnuz_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fnuz
                else:
                    raise ValueError("Dtype not supported for export")
            return quant_metadata
        else:
            return None

    @staticmethod
    def act_quant(x, quant_metadata):
        if quant_metadata is None:
            return x
        scale = quant_metadata['scale']
        zero_point = quant_metadata['zero_point']
        layer_name = quant_metadata['layer_name']
        shared_dict = quant_metadata['shared_dict']
        dtype = quant_metadata['dtype']

        input_quant = StaticScaledQuantizer(
            name=layer_name,
            scale=torch.reciprocal(scale),
            reciprocal_scale=scale,
            offset=zero_point,
            dtype=dtype)
        shared_dict[layer_name] = input_quant
        return x


class SharkLinearQuant(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantLinear

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        self.quant_weight_metadata = self.prepare_weight_for_export(module.weight_quant)
        if self.quant_weight_metadata is not None:
            self.quant_weight_metadata['layer_name'] = self.layer_name + '.weight'
            self.quant_weight_metadata['shared_dict'] = self.shared_dict
        self.quant_input_metadata = self.prepare_act_for_export(module.input_quant)
        if self.quant_input_metadata is not None:
            self.quant_input_metadata['layer_name'] = self.layer_name + '.q_input'
            self.quant_input_metadata['shared_dict'] = self.shared_dict
        self.quant_output_metadata = self.prepare_act_for_export(module.output_quant)
        if self.quant_output_metadata is not None:
            self.quant_output_metadata['layer_name'] = self.layer_name + '.q_output'
            self.quant_output_metadata['shared_dict'] = self.shared_dict
        self.weight = module.weight
        self.bias = module.bias

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        quant_weight = self.weight_quant(self.weight, self.quant_weight_metadata)
        quant_input = self.act_quant(x, self.quant_input_metadata)
        out = torch.nn.functional.linear(quant_input, quant_weight, self.bias)
        quant_out = self.act_quant(out, self.quant_output_metadata)
        return quant_out


class SharkQuantSDPA(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantScaledDotProductAttention

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        self.q_scaled_quant = self.prepare_act_for_export(module.q_scaled_quant.act_quant)
        if self.q_scaled_quant is not None:
            self.q_scaled_quant['layer_name'] = self.layer_name + '_q_output'
            self.q_scaled_quant['shared_dict'] = self.shared_dict
        self.k_transposed_quant = self.prepare_act_for_export(module.k_transposed_quant.act_quant)
        if self.k_transposed_quant is not None:
            self.k_transposed_quant['layer_name'] = self.layer_name + '_k_output'
            self.k_transposed_quant['shared_dict'] = self.shared_dict
        self.v_quant = self.prepare_act_for_export(module.v_quant.act_quant)
        if self.v_quant is not None:
            self.v_quant['layer_name'] = self.layer_name + '_v_output'
            self.v_quant['shared_dict'] = self.shared_dict
        self.pre_forward = module.pre_forward

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            enable_gqa: bool = False):
        assert self.layer_name is not None
        assert self.shared_dict is not None
        query, key, value, _, scale = self.pre_forward(query, key, value, attn_mask, scale, is_causal)

        query = self.act_quant(query, self.q_scaled_quant)
        key = self.act_quant(key, self.q_scaled_quant)
        value = self.act_quant(value, self.q_scaled_quant)

        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa is not None:
            kwargs["enable_gqa"] = enable_gqa

        return torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **kwargs)
