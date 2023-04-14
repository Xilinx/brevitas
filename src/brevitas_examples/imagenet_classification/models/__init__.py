# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from configparser import ConfigParser
import os
from typing import Dict, Optional
from warnings import warn

from torch import hub

from .mobilenetv1 import *
from .proxylessnas import *
from .vgg import *

model_impl = {
    'quant_mobilenet_v1': quant_mobilenet_v1,
    'quant_proxylessnas_mobile14': quant_proxylessnas_mobile14}


def model_with_cfg(name: str, pretrained: Optional[str] = None, bit_width: Optional[int] = None):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './cfg', name + '.ini')
    assert os.path.exists(config_path)
    cfg.read(config_path)
    arch = cfg.get('MODEL', 'ARCH')

    if bit_width is not None:
        cfg.set('QUANT', 'BIT_WIDTH', str(bit_width))

    model = model_impl[arch](cfg)
    if pretrained is not None:
        if pretrained == 'quant_weights':
            weight_url = 'PRETRAINED_URL'
        elif pretrained == 'float_weights':
            weight_url = 'FLOAT_PRETRAINED_URL'
        else:
            raise RuntimeError("Pretrained setting not supported")

        checkpoint = cfg.get('MODEL', weight_url)
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model, cfg


def quant_mobilenet_v1(float_pretrained: bool = True, bit_width: int = 8):
    pretrained = 'float_weights' if float_pretrained else None
    model, _ = model_with_cfg('quant_mobilenet_v1', pretrained=pretrained, bit_width=bit_width)
    return model


def quant_mobilenet_v1_4b(pretrained: bool = True):
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_mobilenet_v1_4b', pretrained)
    return model


def quant_mobilenet_v1_4b_round_avgpool(pretrained: bool = False):
    if pretrained:
        warn(
            "The model was trained with floor TruncAvgPool rather than round,"
            " accuracy will be affected.")
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_mobilenet_v1_4b_round_avgpool', pretrained)
    return model


def quant_proxylessnas_mobile14_4b(pretrained: bool = True):
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_4b', pretrained)
    return model


def quant_proxylessnas_mobile14_4b_round_avgpool(pretrained: bool = False):
    if pretrained:
        warn(
            "The model was trained with floor TruncAvgPool rather than round,"
            " accuracy will be affected.")
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_4b_round_avgpool', pretrained)
    return model


def quant_proxylessnas_mobile14_4b5b(pretrained: bool = True):
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_4b5b', pretrained)
    return model


def quant_proxylessnas_mobile14_hadamard_4b(pretrained: bool = True):
    pretrained = 'quant_weights' if pretrained else None
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_hadamard_4b', pretrained)
    return model
