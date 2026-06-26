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

from brevitas.quant.float_quant_ocp import *
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPAct
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeight
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPAct
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPActPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPActPerTensorFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeight
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerChannelFloat
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerChannelFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e5m2OCPWeightPerTensorFloatMSE
from brevitas.quant.float_quant_ocp import FpOCPAct
from brevitas.quant.float_quant_ocp import FpOCPActPerChannelFloat2d
from brevitas.quant.float_quant_ocp import FpOCPActPerChannelFloat2dMSE
from brevitas.quant.float_quant_ocp import FpOCPActPerTensorFloat
from brevitas.quant.float_quant_ocp import FpOCPActPerTensorFloatMSE
from brevitas.quant.float_quant_ocp import FpOCPMixin
from brevitas.quant.float_quant_ocp import FpOCPWeight
from brevitas.quant.float_quant_ocp import FpOCPWeightPerChannelFloat
from brevitas.quant.float_quant_ocp import FpOCPWeightPerChannelFloatMSE
from brevitas.quant.float_quant_ocp import FpOCPWeightPerTensorFloat
from brevitas.quant.float_quant_ocp import FpOCPWeightPerTensorFloatMSE
