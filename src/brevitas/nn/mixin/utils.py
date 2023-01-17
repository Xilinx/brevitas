# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def filter_kwargs(kwargs_prefix, kwargs: dict):
    return {k[len(kwargs_prefix):]: v for (k, v) in kwargs.items() if k.startswith(kwargs_prefix)}
