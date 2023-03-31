# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def to_0dim_if_scalar(tensor):
    if tensor is not None and len(tensor.shape) == 1 and tensor.shape[0] == 1:
        tensor = tensor.view(())  # 0-Dim tensor
    return tensor


def to_item_if_0dim(tensor):
    if tensor is not None and len(tensor.shape) == 0:
        tensor = tensor.item()
    return tensor
