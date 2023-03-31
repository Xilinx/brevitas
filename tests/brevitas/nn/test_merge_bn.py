# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.nn.utils import merge_bn

SEED = 123456
OUT_CH = 3
IN_CH = 2
FEATURES = 10
KERNEL_SIZE = 3
RTOL = 1e-3


def test_merge_bn():
    torch.manual_seed(SEED)
    conv = torch.nn.Conv2d(IN_CH, OUT_CH, KERNEL_SIZE, bias=True).train(False)
    torch.nn.init.uniform_(conv.weight)
    torch.nn.init.uniform_(conv.bias)
    bn = torch.nn.BatchNorm2d(OUT_CH).train(True)
    input = torch.rand((1, IN_CH, FEATURES, FEATURES))
    bn(conv(input))
    bn.train(False)
    out = bn(conv(input))
    merge_bn(conv, bn)
    out_merged = conv(input)
    all_close = out.isclose(out_merged, rtol=RTOL).all().item()
    assert all_close
