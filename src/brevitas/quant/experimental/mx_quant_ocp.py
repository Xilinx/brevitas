# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "Importing from 'brevitas.quant.experimental.mx_quant_ocp' is deprecated. "
    "Please use 'brevitas.quant.mx_quant_ocp' instead. "
    "Support for importing from the old path will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brevitas.quant.mx_quant_ocp import GroupwiseActFloatProxyMixin
from brevitas.quant.mx_quant_ocp import GroupwiseActProxyMixin
from brevitas.quant.mx_quant_ocp import GroupwiseWeightFloatProxyMixin
from brevitas.quant.mx_quant_ocp import GroupwiseWeightProxyMixin
from brevitas.quant.mx_quant_ocp import MXActMixin
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.mx_quant_ocp import MXInt8WeightMSE
from brevitas.quant.mx_quant_ocp import MXMixin
from brevitas.quant.mx_quant_ocp import MXWeightMixin
from brevitas.quant.mx_quant_ocp import RestrictThresholdMixin
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8Weight
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8WeightMSE
