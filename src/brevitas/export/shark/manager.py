# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from sharktank.types import Dataset
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import Theta
import torch
from torch.nn import Module
import torch.nn as nn

from brevitas.export.inference.manager import _override_act_caching_mode
from brevitas.export.inference.manager import _override_bias_caching_mode
from brevitas.export.inference.manager import _override_weight_caching_mode
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode

from brevitas.export.manager import BaseManager
from brevitas.export.shark.handler import SharkActFloatQuant, SharkWeightFloatQuant
from brevitas.export.shark.handler import SharkActQuant
from brevitas.export.shark.handler import SharkWeightQuant
from brevitas.graph.equalize import EqualizedModule
from brevitas.nn.quant_layer import QuantNonLinearActLayer
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


def _quant_wbiol_handler(layer, layer_name, shared_dict):
    if layer.weight_quant.is_quant_enabled:
        _quant_handler(layer, layer_name, 'weight_quant', shared_dict)
    if layer.input_quant.is_quant_enabled:
        _quant_handler(layer, layer_name, 'input_quant', shared_dict)


def _quant_act_handler(layer, layer_name, shared_dict):
    if layer.act_quant.is_quant_enabled:
        _quant_handler(layer, layer_name, 'act_quant', shared_dict)


def _quant_handler(layer, layer_name, quant_name, shared_dict):
    handler = getattr(layer, quant_name).export_handler
    handler.layer_name = layer_name
    handler.shared_dict = shared_dict


# Inheritance from BaseManager is not techincally needed
class SharkManager(BaseManager):
    handlers = [SharkWeightQuant, SharkActQuant, SharkActFloatQuant, SharkWeightFloatQuant]

    def __init__(self, config=None):
        super().__init__()
        if config == None:
            config = dict()
        self.config = config

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)

    def export(self, model, *model_args, **model_kwargs):

        shared_dict = {}
        # TODO:
        # - Create a base state dict with all layers
        # - Unwrap equalized layer
        # - Flatten equalized layer parameters as part of the "main layer"
        # - Populate layer_name and shared dict fields
        # - ...
        # - Profit (?)

        # sd = model.state_dict()
        # tensors = dict() #{name: DefaultPrimitiveTensor(name=name, data=sd[name]) for name in sd.keys()}

        # shared_dict.update(tensors)

        # Cache quant metadata
        model.apply(lambda m: _override_bias_caching_mode(m, enabled=True, metadata_only=True))
        model.apply(lambda m: _override_act_caching_mode(m, enabled=True))
        model.apply(lambda m: _override_weight_caching_mode(m, enabled=True, metadata_only=False))
        model(*model_args, **model_kwargs)

        model.apply(lambda m: _override_bias_caching_mode(m, enabled=False))
        model.apply(lambda m: _override_act_caching_mode(m, enabled=False))
        model.apply(lambda m: _override_weight_caching_mode(m, enabled=False))

        model.apply(self.set_export_handler)
        self.set_export_mode(model, enabled=True)

        wbiol_id = set()

        for n, m in model.named_modules():
            if isinstance(m, EqualizedModule):
                premul_input = m.scale.weight
                premul_input = DefaultPrimitiveTensor(
                    name=f"{n}.premul_input",
                    data=premul_input,
                )
                shared_dict[premul_input.name] = premul_input
                if isinstance(m.layer, QuantWeightBiasInputOutputLayer):
                    wbiol_id.add(id(m.layer))
                    _quant_wbiol_handler(m.layer, n, shared_dict)
                else: #isinstance(m.layer, torch.nn.Module) and len(list(m.children())) == 0:
                    wbiol_id.add(id(m))
                    for n_p, p in m.layer.named_parameters():
                        param_name = n + '.' + n_p
                        shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)
            if isinstance(m, QuantWeightBiasInputOutputLayer) and id(m) not in wbiol_id:
                wbiol_id.add(id(m))
                _quant_wbiol_handler(m, n, shared_dict)

            elif isinstance(m, QuantNonLinearActLayer):
                _quant_act_handler(m, n, shared_dict)
            elif isinstance(m, torch.nn.Module) and len(list(m.children())) == 0 and id(m) not in wbiol_id:
                for n_p, p in m.named_parameters():
                    param_name = n + '.' + n_p
                    shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)

        model(*model_args, **model_kwargs)

        self.set_export_mode(model, enabled=False)

        theta = Theta(shared_dict)
        theta = theta.flatten()
        ds = Dataset(self.config, Theta(theta))

        return ds
