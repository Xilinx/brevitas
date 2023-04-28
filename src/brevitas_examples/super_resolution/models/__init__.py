# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn

from .espcn import *

model_impl = {
    'float_espcn': float_espcn,
    'quant_espcn_w8a8': quant_espcn_w8a8,
    'quant_espcn_w4a4': quant_espcn_w4a4,
    'quant_espcn_finn_a2q_w4a4_14b': quant_espcn_finn_a2q_w4a4_14b,
    'quant_espcn_finn_a2q_w4a4_32b': quant_espcn_finn_a2q_w4a4_32b}


def get_model_by_name(name: str, upscale_factor: int) -> nn.Module:
    if name not in model_impl.keys():
        raise NotImplementedError(f"{name} does not exist.")
    model = model_impl[name](upscale_factor)
    return model
