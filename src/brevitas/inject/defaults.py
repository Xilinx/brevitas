from dependencies import this

from . import BaseInjector as Injector
from .enum import QuantType
from .base import *


class FloatBias(Injector):
    """

    """
    quant_type = QuantType.FP
    narrow_range = False
    signed = True


class TruncTo8bit(IntTrunc):
    """

    """
    bit_width = 8


class Int8Bias(IntQuant):
    """

    """
    bit_width = 8


class Int8BiasPerTensorFloatInternalScaling(
    IntQuant, MaxStatsScaling, PerTensorFloatScaling8bit):
    """

    """
    pass


class Int8WeightPerTensorFloat(
    NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit):
    """

    """
    pass


class ShiftedUint8WeightPerTensorFloat(
    ShiftedMinUintQuant, MinMaxStatsScaling, PerTensorFloatScaling8bit):
    """

    """
    pass


class Int8ActPerTensorFloat(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit):
    """
    """
    pass


class Uint8ActPerTensorFloat(
    UintQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit):
    """

    """
    pass


class ShiftedUint8ActPerTensorFloat(
    ShiftedIntToUintQuant, ParamFromRuntimeMinMaxScaling, PerTensorFloatScaling8bit):
    """

    """
    pass


class Int8ActPerTensorFloatMinMaxInit(
    IntQuant & ParamMinMaxInitScaling & PerTensorFloatScaling8bit):
    """
    """
    pass