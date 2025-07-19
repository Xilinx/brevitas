# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch.distributed as dist


def init_process_group(backend: str = "nccl") -> None:
    # Verify if the script was launched with torch_elastic
    if dist.is_torchelastic_launched():
        # If that is the case, initialize the default process group
        dist.init_process_group(backend=backend)
