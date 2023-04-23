# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from .espcn import *

model_impl = {
    'quant_espcn_x3_v1_4b': quant_espcn_x3_v1_4b,
    'quant_espcn_x3_v2_4b': quant_espcn_x3_v2_4b,
    'quant_espcn_x3_v3_4b': quant_espcn_x3_v3_4b}


def get_model_by_name(name):
    model = model_impl[name]()
    return model
