# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "Importing from 'brevitas.quant.experimental.float_base' is deprecated. "
    "Please use 'brevitas.quant.float_base' instead. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brevitas.quant.float_base import *
from brevitas.quant.float_base import FloatActBase
from brevitas.quant.float_base import FloatBase
from brevitas.quant.float_base import FloatWeightBase
from brevitas.quant.float_base import Fp4e2m1Mixin
from brevitas.quant.float_base import Fp6e2m3Mixin
from brevitas.quant.float_base import Fp6e3m2Mixin
from brevitas.quant.float_base import Fp8e4m3Mixin
from brevitas.quant.float_base import Fp8e5m2Mixin
from brevitas.quant.float_base import ScaledFloatActBase
from brevitas.quant.float_base import ScaledFloatWeightBase
