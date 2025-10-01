# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from functools import partial
from functools import reduce
import itertools
from unittest.mock import patch

from packaging.version import parse
import pytest
import torch
import torch.nn.utils.parametrize as parametrize
from torchvision import models

from brevitas import torch_version
from brevitas.fx import symbolic_trace
from brevitas.graph.base import ModuleInstanceRegisterParametrization
from brevitas.graph.equalize import _apply_had_device
from brevitas.graph.equalize import _apply_ort_device
from brevitas.graph.equalize import _apply_rotate
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizationIndexes
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import MergeLnAffine
from brevitas.graph.equalize import random_orthogonal_matrix
from brevitas.graph.equalize import Region
from brevitas.graph.equalize import rotate_permute_mode
from brevitas.graph.hadamard import get_hadK
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from tests.marker import requires_pt_ge

from .equalization_fixtures import *


class ConvGroupConvModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Embedding(3, 8)
        self.conv_0 = nn.Embedding(3, 8)
        self.conv_1 = nn.Linear(8, 8)
        self.relu = nn.SiLU()

    def forward(self, x):
        start = x
        x = self.conv(start)
        x_0 = self.conv_0(start)
        x = self.relu(x)
        x = x * x_0
        x = self.conv_1(x)
        return x


def test_rotation_permute():
    inp = torch.LongTensor([[0, 1, 2, 2], [2, 1, 0, 1]])
    model = ConvGroupConvModel()
    model(inp)
    model, _ = torch._dynamo.export(model)(inp)
    o = model(inp)
    with rotate_permute_mode(model, orphan_sink=True, apply_permute=True, return_rewriters=True):
        model(inp)
    o1 = model(inp)

    assert torch.allclose(o, o1)
