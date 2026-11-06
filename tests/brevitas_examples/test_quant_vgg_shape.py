# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""QuantVGG classifier in_features must match TruncAvgPool2d output (#1500)."""

import torch

from brevitas_examples.imagenet_classification.models.vgg import quant_vgg11


def test_quant_vgg11_224_forward_shape():
    # Five MaxPool2d layers map 224x224 -> 7x7 spatial before avgpool.
    # TruncAvgPool2d(kernel_size=7, stride=1) then yields 1x1 -> flatten size 512.
    model = quant_vgg11(bit_width=8, num_classes=1000)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1000)
