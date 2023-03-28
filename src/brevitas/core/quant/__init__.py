# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# retrocompatibility
# the assert prevents the removal of the unused import
from brevitas.inject.enum import QuantType

assert QuantType

from .binary import BinaryQuant
from .binary import ClampedBinaryQuant
from .int import PrescaledRestrictIntQuant
from .int import PrescaledRestrictIntQuantWithInputBitWidth
from .int import RescalingIntQuant
from .int import TruncIntQuant
from .int_base import DecoupledIntQuant
from .int_base import IntQuant
from .ternary import TernaryQuant
