# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn

from brevitas.graph.utils import get_module_name_and_parent
from brevitas.graph.utils import set_module
from tests.conftest import SEED

IN_FEATURES = 2
OUT_FEATURES = 2

torch.manual_seed(SEED)


class SubModel(nn.Module):

    def __init__(self):
        super(SubModel, self).__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.sub_model = SubModel()

    def forward(self, x):
        return self.sub_model(x)


def test_get_module_name_and_parent():
    model = Model()
    module_name, supermodule = get_module_name_and_parent(model, "sub_model.linear")
    assert module_name == "linear"
    assert supermodule is model.sub_model


def test_set_module():
    model = Model()
    new_module = nn.Linear(IN_FEATURES, OUT_FEATURES)
    set_module(model, new_module, "sub_model.linear")
    assert model.sub_model.linear is new_module
