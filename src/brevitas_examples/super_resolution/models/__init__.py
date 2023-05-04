# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn
from functools import partial

from .espcn import *

model_impl = {
    'float_espcn_x2': partial(float_espcn, upscale_factor=2),
    'quant_espcn_x2_w8a8_base': partial(quant_espcn_base, upscale_factor=2, weight_bit_width=8, act_bit_width=8),
    'quant_espcn_x2_w8a8_a2q_32b': partial(quant_espcn_a2q, upscale_factor=2, weight_bit_width=8, act_bit_width=8, acc_bit_width=32),
    'quant_espcn_x2_w8a8_a2q_16b': partial(quant_espcn_a2q, upscale_factor=2, weight_bit_width=8, act_bit_width=8, acc_bit_width=16)}


def get_model_by_name(name: str) -> nn.Module:
    if name not in model_impl.keys():
        raise NotImplementedError(f"{name} does not exist.")
    model = model_impl[name]()
    return model
