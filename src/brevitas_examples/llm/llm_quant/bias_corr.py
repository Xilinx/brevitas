"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.calibrate import bias_correction_mode


@torch.no_grad()
def apply_bias_correction(model, dataloader):
    with bias_correction_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)
