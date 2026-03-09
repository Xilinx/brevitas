# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "Importing from 'brevitas.quant.experimental.float_quant_ocp' is deprecated. "
    "Please use 'brevitas.quant.float_quant_ocp' instead. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brevitas.quant.float_quant_ocp import *  # noqa: F401, F403, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPAct  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeight  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPAct  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeight  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPAct  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPActPerChannelFloat2d  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPActPerChannelFloat2dMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPActPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPActPerTensorFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPMixin  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPWeight  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPWeightPerChannelFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPWeightPerChannelFloatMSE  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPWeightPerTensorFloat  # noqa: F401, E402
from brevitas.quant.float_quant_ocp import FpOCPWeightPerTensorFloatMSE  # noqa: F401, E402
