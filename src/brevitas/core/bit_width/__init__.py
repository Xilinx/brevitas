# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from .const import BitWidthConst, MsbClampBitWidth
from .parameter import BitWidthParameter, RemoveBitwidthParameter

# retrocompatibility
# the assert prevents the removal of the unused import
from brevitas.inject.enum import BitWidthImplType
assert BitWidthImplType