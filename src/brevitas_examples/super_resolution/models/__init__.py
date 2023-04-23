# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from .espcn import *

model_impl = {
    'quant_espcn_v1_4b': quant_espcn_v1}


def model_with_cfg(name, pretrained):
    num_channels = 1
    upscale_factor = 3
    weight_bit_width = 4
    act_bit_width = 4
    model = model_impl[name](
        num_channels=num_channels,
        upscale_factor=upscale_factor,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width)
    return model, None


def quant_espcn_v1_4b():
    print("done")
