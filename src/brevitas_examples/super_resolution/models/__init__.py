# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Union

from torch import hub

from .espcn import *

model_impl = {
    'float_espcn_x2':
        partial(float_espcn, upscale_factor=2),
    'quant_espcn_x2_w8a8_base':
        partial(quant_espcn_base, upscale_factor=2, weight_bit_width=8, act_bit_width=8),
    'quant_espcn_x2_w8a8_a2q_32b':
        partial(
            quant_espcn_a2q,
            upscale_factor=2,
            weight_bit_width=8,
            act_bit_width=8,
            acc_bit_width=32),
    'quant_espcn_x2_w8a8_a2q_16b':
        partial(
            quant_espcn_a2q,
            upscale_factor=2,
            weight_bit_width=8,
            act_bit_width=8,
            acc_bit_width=16)}

root_url = 'https://github.com/Xilinx/brevitas/releases/download/super_res-r0'

model_url = {
    'float_espcn_x2': f'{root_url}/float_espcn_x2-758c8fef.pth',
    'quant_espcn_x2_w8a8_base': f'{root_url}/quant_espcn_x2_w8a8_base-7d54e29c.pth',
    'quant_espcn_x2_w8a8_a2q_32b': f'{root_url}/quant_espcn_x2_w8a8_a2q_32b-0b1f361d.pth',
    'quant_espcn_x2_w8a8_a2q_16b': f'{root_url}/quant_espcn_x2_w8a8_a2q_16b-3c4acd35.pth',}


def get_model_by_name(name: str, pretrained: bool = False) -> Union[FloatESPCN, QuantESPCN]:
    if name not in model_impl.keys():
        raise NotImplementedError(f"{name} does not exist.")
    model = model_impl[name]()
    if pretrained:
        checkpoint = model_url[name]
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model
