# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# retrocompatibility
# the assert prevents the removal of the unused import
from brevitas.inject.enum import BitWidthImplType

from .const import BitWidthConst
from .const import BitWidthStatefulConst
from .const import MsbClampBitWidth
from .parameter import BitWidthParameter
from .parameter import RemoveBitwidthParameter

assert BitWidthImplType
