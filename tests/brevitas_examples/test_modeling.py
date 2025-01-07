# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Dict, Optional
import unittest

import torch
import torch.nn as nn

from brevitas_examples.common.accelerate_utils.modeling import infer_auto_device_map


class ModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class ModelingUtilsTester(unittest.TestCase):

    def test_infer_auto_device_map_tied_weights_split(self):
        model = nn.Sequential(OrderedDict([("layer1", ModelForTest()), ("layer2", ModelForTest())]))
        expected_sizes = {"": 236, "linear1": 64, "linear1.weight": 48, "linear1.bias": 16}
        expected_sizes.update({"linear2": 100, "linear2.weight": 80, "linear2.bias": 20})
        expected_sizes.update({"batchnorm": 72, "batchnorm.weight": 16, "batchnorm.bias": 16})
        expected_sizes.update({
            "batchnorm.running_mean": 16,
            "batchnorm.running_var": 16,
            "batchnorm.num_batches_tracked": 8})
        # model has size 236: linear1 64, batchnorm 72, linear2 100
        model.layer1.linear1.weight = model.layer2.linear1.weight
        device_map = infer_auto_device_map(
            model,
            max_memory={
                0: 236, 1: 236, 2: 236},
            verbose=True,
            no_split_module_classes=[ModelForTest.__name__])
        assert device_map == {"layer1": 1, "layer2": 2}
