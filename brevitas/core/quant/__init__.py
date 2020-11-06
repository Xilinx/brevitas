# retrocompatibility
# the assert prevents the removal of the unused import
from brevitas.inject.enum import QuantType
assert QuantType

from .binary import ClampedBinaryQuant, BinaryQuant
from .ternary import TernaryQuant
from .int_base import IntQuant
from .int import TruncIntQuant, RescalingIntQuant, PrescaledRestrictIntQuant
from .int import PrescaledRestrictIntQuantWithInputBitWidth
from .placeholder import IdentityPrescaledQuant, IdentityQuant
