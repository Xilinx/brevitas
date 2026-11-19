# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def register():
    """vLLM general plugin entry point, executed in every vLLM process."""
    # Importing the manager registers the quant_brevitas quantization method
    import brevitas.export.inference.vLLM.manager
