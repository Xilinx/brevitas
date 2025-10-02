# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.graph.equalize import rotate_permute_mode

from .equalization_fixtures import *


class ConvGroupConvModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Embedding(3, 32)
        self.conv_0 = nn.Embedding(3, 32)
        self.conv_1 = nn.Linear(32, 32)
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
