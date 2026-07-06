# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "The 'brevitas.quant.experimental' package is deprecated. "
    "Quantizers have been moved to 'brevitas.quant'. "
    "Please update your imports accordingly. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
