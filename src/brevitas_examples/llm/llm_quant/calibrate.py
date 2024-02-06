"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode


@torch.no_grad()
def apply_calibration(model, dataloader):
    with calibration_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)
