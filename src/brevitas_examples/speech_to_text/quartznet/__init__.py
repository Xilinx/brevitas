# Adapted from https://github.com/NVIDIA/NeMo/blob/r0.9/collections/nemo_asr/
# Copyright (C) 2020 Xilinx (Giuseppe Franco)
# Copyright (C) 2019 NVIDIA CORPORATION.
#
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .data_layer import (
        AudioToTextDataLayer)
from .greedy_ctc_decoder import GreedyCTCDecoder
from .quartznet import quartznet
from .losses import CTCLossNM
from .helpers import *

import os
from configparser import ConfigParser
from ruamel.yaml import YAML
from torch import hub

__all__ = ['AudioToTextDataLayer',
           'quartznet',
           'quant_quartznet_perchannelscaling_4b',
           'quant_quartznet_perchannelscaling_8b',
           'quant_quartznet_pertensorscaling_8b']

name = "quarznet_release"
model_impl = {
    'quartznet': quartznet,
}


def model_with_cfg(name, pretrained, export_mode):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'cfg', name + '.ini')
    assert os.path.exists(config_path)
    cfg.read(config_path)
    arch = cfg.get('MODEL', 'ARCH')
    topology_file = cfg.get('MODEL', 'TOPOLOGY_FILE')
    topology_path = os.path.join(current_dir, '..', 'cfg', 'topology', topology_file)
    yaml = YAML(typ="safe")
    with open(topology_path) as f:
        quartnzet_params = yaml.load(f)
    model = model_impl[arch](cfg, quartnzet_params, export_mode)
    if pretrained:
        pretrained_encoder_url = cfg.get('MODEL', 'PRETRAINED_ENCODER_URL')
        pretrained_decoder_url = cfg.get('MODEL', 'PRETRAINED_DECODER_URL')
        print("=> Loading encoder checkpoint from:'{}'".format(pretrained_encoder_url))
        print("=> Loading decoder checkpoint from:'{}'".format(pretrained_decoder_url))
        checkpoint_enc = torch.hub.load_state_dict_from_url(pretrained_encoder_url, progress=True, map_location='cpu')
        checkpoint_dec = torch.hub.load_state_dict_from_url(pretrained_decoder_url, progress=True, map_location='cpu')
        model.restore_checkpoints(checkpoint_enc, checkpoint_dec)
    return model, cfg


def quant_quartznet_perchannelscaling_4b(pretrained=True, export_mode=False):
    model, _ = model_with_cfg('quant_quartznet_perchannelscaling_4b', pretrained, export_mode)
    return model


def quant_quartznet_perchannelscaling_8b(pretrained=True, export_mode=False):
    model, _ = model_with_cfg('quant_quartznet_perchannelscaling_8b', pretrained, export_mode)
    return model


def quant_quartznet_pertensorscaling_8b(pretrained=True, export_mode=False):
    model, _ = model_with_cfg('quant_quartznet_pertensorscaling_8b', pretrained, export_mode)
    return model
