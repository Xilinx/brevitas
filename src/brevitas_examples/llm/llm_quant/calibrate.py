"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode


class groupwise_calibration_mode(calibration_mode):
    def __init__(self, model):
        super(calibration_mode, self).__init__(
            model=model,
            call_act_quantizer_impl=True,
            disable_act_quant=False,
            disable_weight_quant=True,
            disable_bias_quant=True,
            is_training=True)
        self.enabled = True

@torch.no_grad()
def apply_calibration(model, dataloader):
    model.train()
    with groupwise_calibration_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)
    model.eval()
