# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from builtins import staticmethod
from functools import reduce
from typing import Any, List

import torch
import torch.distributed as dist


class TensorBucket:

    def __init__(self, flattened_tensors: List[torch.Tensor], shapes: List[torch.Size]) -> None:
        self.flattened_tensor = torch.cat(flattened_tensors)
        self.shapes = shapes

    @staticmethod
    def bucketize_tensors(tensors: List[torch.Tensor], bucket_size: int = 1e8) -> "TensorBucket":
        flattened_tensors_bucket, shapes_bucket = [], []
        curr_bucket_size = 0
        for tensor in tensors:
            # Check if tensor fits in the current bucket
            if curr_bucket_size + tensor.numel() * tensor.element_size() > bucket_size:
                # If not, create a new bucket
                yield TensorBucket(flattened_tensors=flattened_tensors_bucket, shapes=shapes_bucket)
                curr_bucket_size = 0
                flattened_tensors_bucket, shapes_bucket = [], []
            # Tensor fits in the current bucket
            flattened_tensors_bucket.append(tensor.view(-1))
            shapes_bucket.append(tensor.shape)
        # Create remaining bucket
        yield TensorBucket(flattened_tensors=flattened_tensors_bucket, shapes=shapes_bucket)

    def debucketize_tensors(self) -> torch.Tensor:
        offset = 0
        for shape in self.shapes:
            n_element = reduce(lambda x, y: x * y, shape)
            yield self.flattened_tensor[offset:offset + n_element].view(shape)
            offset += n_element


def init_process_group(backend: str = "nccl") -> None:
    # Verify if the script was launched with torch_elastic
    if dist.is_torchelastic_launched():
        # If that is the case, initialize the default process group
        dist.init_process_group(backend=backend)
