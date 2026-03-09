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

from brevitas.quant.float_base import *  # noqa: F401, F403, E402
from brevitas.quant.float_base import FloatActBase  # noqa: F401, E402
from brevitas.quant.float_base import FloatBase  # noqa: F401, E402
from brevitas.quant.float_base import FloatWeightBase  # noqa: F401, E402
from brevitas.quant.float_base import Fp4e2m1Mixin  # noqa: F401, E402
from brevitas.quant.float_base import Fp6e2m3Mixin  # noqa: F401, E402
from brevitas.quant.float_base import Fp6e3m2Mixin  # noqa: F401, E402
from brevitas.quant.float_base import Fp8e4m3Mixin  # noqa: F401, E402
from brevitas.quant.float_base import Fp8e5m2Mixin  # noqa: F401, E402
from brevitas.quant.float_base import ScaledFloatActBase  # noqa: F401, E402
from brevitas.quant.float_base import ScaledFloatWeightBase  # noqa: F401, E402
