# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from pathlib import Path

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
from brevitas.export.shark.handler import SharkActFloatQuant
from brevitas.export.shark.handler import SharkActQuant
from brevitas.export.shark.handler import SharkWeightFloatQuant
from brevitas.export.shark.handler import SharkWeightQuant
from brevitas.graph.equalize import EqualizedModule
from brevitas.nn.quant_layer import QuantNonLinearActLayer
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


def _quant_wbiol_handler(layer, layer_name, shared_dict):
    if layer.weight_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.weight', 'weight_quant', shared_dict)
    else:
        shared_dict[layer_name + '.' + 'weight'] = DefaultPrimitiveTensor(name=layer_name + '.' + 'weight', data=layer.weight)
    if layer.input_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.q_input', 'input_quant', shared_dict)
    if layer.output_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.q_output', 'output_quant', shared_dict)


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

    def gguf_preprocess(self, model):
        import sys

        import gguf

        from brevitas_examples.llm.gguf_export.convert import ModelBase
        """Export the model to gguf format."""
        output_type = gguf.LlamaFileType.ALL_F32

        config = model.config

        tmp_work_dir = Path('./tmp_dir')
        config.save_pretrained(tmp_work_dir)

        with torch.no_grad():
            hparams = ModelBase.load_hparams(tmp_work_dir)
            model_architecture = hparams["architectures"][0]
            try:
                model_class = ModelBase.from_model_architecture(model_architecture)
            except NotImplementedError:
                sys.exit(1)
            model_class = ModelBase.from_model_architecture(model_architecture)
            model_name = model.name_or_path.split('/')
            if len(model_name[-1]) == 0:
                model_name = model_name[-2]
            else:
                model_name = model_name[-1]

            model_instance = model_class(
                model,
                dir_model=tmp_work_dir,
                ftype=output_type,
                fname_out=tmp_work_dir,
                is_big_endian=False,
                model_name=model_name,
                split_max_tensors=False,
                split_max_size=0,
                dry_run=False,
                small_first_shard=False)
        return model_instance

    def export(self, model, *model_args, **model_kwargs):
        from brevitas_examples.llm.main import offload_model
        from brevitas_examples.llm.main import remove_hooks

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
        # model.apply(lambda m: _override_bias_caching_mode(m, enabled=True, metadata_only=True))
        # model.apply(lambda m: _override_act_caching_mode(m, enabled=True))
        # model.apply(lambda m: _override_weight_caching_mode(m, enabled=True, metadata_only=False))
        # offload_model(model)
        # with torch.no_grad():
        #     model(*model_args, **model_kwargs)
        # remove_hooks(model)
        # model.apply(lambda m: _override_bias_caching_mode(m, enabled=False))
        # model.apply(lambda m: _override_act_caching_mode(m, enabled=False))
        # model.apply(lambda m: _override_weight_caching_mode(m, enabled=False))

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
                else:  #isinstance(m.layer, torch.nn.Module) and len(list(m.children())) == 0:
                    wbiol_id.add(id(m))
                    for n_p, p in m.layer.named_parameters():
                        param_name = n + '.' + n_p
                        shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)
            if isinstance(m, QuantWeightBiasInputOutputLayer) and id(m) not in wbiol_id:
                wbiol_id.add(id(m))
                _quant_wbiol_handler(m, n, shared_dict)

            elif isinstance(m, QuantNonLinearActLayer):
                _quant_act_handler(m, n, shared_dict)
            elif isinstance(m, torch.nn.Module) and len(list(
                    m.children())) == 0 and id(m) not in wbiol_id:
                for n_p, p in m.named_parameters():
                    param_name = n + '.' + n_p
                    shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)
        print("Forward starts")
        offload_model(model)
        with torch.no_grad():
            model(*model_args, **model_kwargs)
        remove_hooks(model)
        print("Forward ends")

        self.set_export_mode(model, enabled=False)

        updated_theta = dict()
        gguf_model = self.gguf_preprocess(model)
        for k, v in shared_dict.items():
            if k.endswith('.q_input'):
                prefix = k.removesuffix('.q_input')
                suffix = '.q_input'
            elif k.endswith('.q_output'):
                prefix = k.removesuffix('.q_output')
                suffix = '.q_output'
            else:
                prefix = k
                suffix = ''
            new_k = gguf_model.map_tensor_name(prefix)
            v.name = new_k + suffix
            updated_theta[new_k + suffix] = v
        ds = Dataset(self.config, Theta(updated_theta))
        ds.save('test_dataset.irpa', io_report_callback=None)
        return ds
