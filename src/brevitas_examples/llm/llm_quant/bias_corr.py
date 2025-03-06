"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from brevitas.graph.calibrate import bias_correction_mode
from brevitas_examples.llm.llm_quant.data_utils import DatasetToDevice


# Function to batchify the dataset
def collate_fn(kwargs_list, return_tensors="pt"):
    kwargs = {}
    for curr_dict in kwargs_list:
        for key, value in curr_dict.items():
            if isinstance(value, torch.Tensor):
                if key not in kwargs:
                    kwargs[key] = []
                kwargs[key].append(value)
            else:
                if key not in kwargs:
                    kwargs[key] = value
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) > 0:
            kwargs[key] = torch.cat(kwargs[key], dim=0)
    return kwargs


def _maybe_partition_dataloader(dataloader: DatasetToDevice) -> DatasetToDevice:
    # If multiple processes are running simultaneously, each receives a different partition
    if dist.is_initialized():
        rank = dist.get_rank()
        partition_size = len(dataloader) // dist.get_world_size()
        dataloader = DatasetToDevice(
            dataloader.data[rank * partition_size:(rank + 1) * partition_size], dataloader.device)
    return dataloader


@torch.no_grad()
def apply_bias_correction(model, dataloader, batch_size=1):
    dataloader = _maybe_partition_dataloader(dataloader)
    bias_correction_dataloader = DataLoader(
        dataloader, collate_fn=collate_fn, batch_size=batch_size)
    with bias_correction_mode(model, batch_size=batch_size):
        for inps in tqdm(dataloader):
            model(**inps)
