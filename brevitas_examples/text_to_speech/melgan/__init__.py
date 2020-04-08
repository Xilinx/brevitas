import os
from configparser import ConfigParser

from torch import hub

from .melgan import *

model_impl = {
    'melgan': melgan,
}


def model_with_cfg(name, pretrained):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'cfg', name + '.ini')
    assert os.path.exists(config_path)
    cfg.read(config_path)
    arch = cfg.get('MODEL', 'ARCH')
    model = model_impl[arch](cfg)
    if pretrained:
        checkpoint = cfg.get('MODEL', 'PRETRAINED_URL')
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model, cfg


def quant_melgan_8b(pretrained=True):
    model, _ = model_with_cfg('quant_melgan_8b', pretrained)
    return model
