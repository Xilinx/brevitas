# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gguf
from sharktank.types import Dataset
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import Theta
import torch
from torch.nn import Module

from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.shark.handler import SharkActEqualization
from brevitas.export.shark.handler import SharkLinearQuant
from brevitas.export.shark.handler import SharkQuantSDPA


# Inheritance from BaseManager is not techincally needed
class SharkManager(BaseManager):
    handlers = [SharkActEqualization, SharkLinearQuant, SharkQuantSDPA]

    def __init__(self, config=None):
        super().__init__()
        if config == None:
            config = dict()
        self.config = config

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module, name: str, shared_dict: dict):
        _set_layer_export_handler(cls, module)
        if hasattr(module, 'export_handler') and module.export_handler is not None:
            module.export_handler.layer_name = name
            module.export_handler.shared_dict = shared_dict

    def export(self, model, *model_args, **model_kwargs):

        shared_dict = {}

        for name, module in model.named_modules():
            self.set_export_handler(module, name, shared_dict)
        self.set_export_mode(model, enabled=True)

        with torch.no_grad():
            model(*model_args, **model_kwargs)

        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Module) and len(list(m.children())) == 0:
                for n_p, p in m.named_parameters():
                    param_name = n + '.' + n_p
                    if param_name in shared_dict:
                        continue
                    shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)

        self.set_export_mode(model, enabled=False)

        theta = Theta(shared_dict)
        ds = Dataset(self.config.to_dict(), theta)
        return ds
