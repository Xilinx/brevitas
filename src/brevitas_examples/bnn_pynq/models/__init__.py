# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os
from configparser import ConfigParser

import torch
from torch import hub

__all__ = ['cnv_1w1a', 'cnv_1w2a', 'cnv_2w2a',
           'sfc_1w1a', 'sfc_1w2a', 'sfc_2w2a',
           'tfc_1w1a', 'tfc_1w2a', 'tfc_2w2a',
           'lfc_1w1a', 'lfc_1w2a',
           'model_with_cfg']

from .CNV import cnv
from .FC import fc

model_impl = {
    'CNV': cnv,
    'FC': fc,
}

def get_model_cfg(name):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'cfg', name.lower() + '.ini')
    assert os.path.exists(config_path)
    cfg.read(config_path)
    return cfg

def model_with_cfg(name, pretrained):
    cfg = get_model_cfg(name)
    arch = cfg.get('MODEL', 'ARCH')
    model = model_impl[arch](cfg)
    if pretrained:
        checkpoint = cfg.get('MODEL', 'PRETRAINED_URL')
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model, cfg


def cnv_1w1a(pretrained=True):
    model, _ = model_with_cfg('cnv_1w1a', pretrained)
    return model


def cnv_1w2a(pretrained=True):
    model, _ = model_with_cfg('cnv_1w2a', pretrained)
    return model


def cnv_2w2a(pretrained=True):
    model, _ = model_with_cfg('cnv_2w2a', pretrained)
    return model


def sfc_1w1a(pretrained=True):
    model, _ = model_with_cfg('sfc_1w1a', pretrained)
    return model


def sfc_1w2a(pretrained=True):
    model, _ = model_with_cfg('sfc_1w2a', pretrained)
    return model


def sfc_2w2a(pretrained=True):
    model, _ = model_with_cfg('sfc_2w2a', pretrained)
    return model


def tfc_1w1a(pretrained=True):
    model, _ = model_with_cfg('tfc_1w1a', pretrained)
    return model


def tfc_1w2a(pretrained=True):
    model, _ = model_with_cfg('tfc_1w2a', pretrained)
    return model


def tfc_2w2a(pretrained=True):
    model, _ = model_with_cfg('tfc_2w2a', pretrained)
    return model


def lfc_1w1a(pretrained=True):
    model, _ = model_with_cfg('lfc_1w1a', pretrained)
    return model


def lfc_1w2a(pretrained=True):
    model, _ = model_with_cfg('lfc_1w2a', pretrained)
    return model


