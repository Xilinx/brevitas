from . import BaseInjector as Injector


class DefaultWeightScalingInjector(Injector):
    scaling_impl_type = 'STATS'
    restrict_scaling_type = 'FP'
    scaling_stats_op = 'MAX'
    scaling_per_output_channel = False
    scaling_min_val = 2.0 ** (-16)


class DefaultWeightQuantInjector(DefaultWeightScalingInjector):
    quant_type = 'INT'
    bit_width_impl_type = 'CONST'
    narrow_range = True
    signed = True
    bit_width = 8


class DefaultBiasQuantInjector(Injector):
    quant_type = 'FP'
    narrow_range = False
    signed = True


class DefaultActQuantInjector(Injector):
    quant_type = 'INT'
    bit_width_impl_type = 'CONST'
    bit_width = 8
    scaling_impl_type = 'PARAMETER'
    restrict_scaling_type = 'LOG_FP'
    scaling_per_output_channel = False
    scaling_min_val = 2.0 ** (-16)


class DefaultTruncQuantInjector(Injector):
    quant_type = 'INT'
    bit_width_impl_type = 'CONST'
    float_to_int_impl_type = 'FLOOR'
    bit_width = 8


class DefaultSignedActQuantInjector(DefaultActQuantInjector):
    signed = True
    narrow_range = False


class DefaultUnsignedActQuantInjector(DefaultActQuantInjector):
    signed = False
    narrow_range = False
    min_val = 0.0


class DefaultUnitarySignedActQuantInjector(DefaultSignedActQuantInjector):
    min_val = -1.0
    max_val = 1.0


class DefaultUnitaryUnsignedActQuantInjector(DefaultUnsignedActQuantInjector):
    max_val = 1.0
