import pytest
import torch
from torch import Tensor
from torchvision import models

from brevitas.nn.utils import merge_bn

SEED = 123456
OUT_CH = 32
IN_CH = 16
FEATURES = 8
KERNEL_SIZE = 3
ATOL = 1e-2

def test_merge_bn():
    conv = torch.nn.Conv2d(IN_CH, OUT_CH, KERNEL_SIZE).train(False)
    bn = torch.nn.BatchNorm2d(OUT_CH).train(True)
    input = torch.randn((1, IN_CH, FEATURES, FEATURES))
    bn(conv(input))
    bn.train(False)
    out = bn(conv(input))
    merge_bn(conv, bn, affine_only=False)
    out_merged = conv(input)
    all_close = out.isclose(out_merged, atol=ATOL).all().item()
    assert all_close