# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "Importing from 'brevitas.quant.experimental.float_quant_fnuz' is deprecated. "
    "Please use 'brevitas.quant.float_quant_fnuz' instead. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brevitas.quant.float_quant_fnuz import *
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZAct
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloatMSE
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeight
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloatMSE
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloatMSE
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZAct
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZActPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZActPerTensorFloatMSE
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeight
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerChannelFloat
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerChannelFloatMSE
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerTensorFloatMSE
from brevitas.quant.float_quant_fnuz import FpFNUZAct
from brevitas.quant.float_quant_fnuz import FpFNUZActPerChannelFloat2d
from brevitas.quant.float_quant_fnuz import FpFNUZActPerChannelFloat2dMSE
from brevitas.quant.float_quant_fnuz import FpFNUZActPerTensorFloat
from brevitas.quant.float_quant_fnuz import FpFNUZActPerTensorFloatMSE
from brevitas.quant.float_quant_fnuz import FpFNUZMixin
from brevitas.quant.float_quant_fnuz import FpFNUZWeight
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerChannelFloat
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerChannelFloatMSE
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerTensorFloat
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerTensorFloatMSE
