# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "Importing from 'brevitas.quant.experimental.float' is deprecated. "
    "Please use 'brevitas.quant.float' instead. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brevitas.quant.float import Fp8e4m3Act
from brevitas.quant.float import Fp8e4m3ActPerChannelFloat2d
from brevitas.quant.float import Fp8e4m3ActPerChannelFloat2dMSE
from brevitas.quant.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.float import Fp8e4m3ActPerTensorFloatMSE
from brevitas.quant.float import Fp8e4m3Weight
from brevitas.quant.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.float import Fp8e4m3WeightPerChannelFloatMSE
from brevitas.quant.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.float import Fp8e4m3WeightPerTensorFloatMSE
from brevitas.quant.float import Fp8e5m2Act
from brevitas.quant.float import Fp8e5m2ActPerChannelFloat2d
from brevitas.quant.float import Fp8e5m2ActPerChannelFloat2dMSE
from brevitas.quant.float import Fp8e5m2ActPerTensorFloat
from brevitas.quant.float import Fp8e5m2ActPerTensorFloatMSE
from brevitas.quant.float import Fp8e5m2Weight
from brevitas.quant.float import Fp8e5m2WeightPerChannelFloat
from brevitas.quant.float import Fp8e5m2WeightPerTensorFloat
