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

from brevitas.quant.float_quant_fnuz import *  # noqa: F401, F403, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZAct  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeight  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZAct  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeight  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import Fp8e5m2FNUZWeightPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZAct  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZActPerChannelFloat2d  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZActPerChannelFloat2dMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZMixin  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZWeight  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_fnuz import FpFNUZWeightPerTensorFloatMSE  # noqa: F401, E402
