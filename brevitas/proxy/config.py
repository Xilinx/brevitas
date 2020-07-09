# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from functools import partial
from dependencies import Injector

from brevitas.core.quant import *
from brevitas.core.function_wrapper import RoundSte, CeilSte, FloorSte, TensorClamp, TensorClampSte
from brevitas.core.scaling import *
from brevitas.core.restrict_val import *

from brevitas.core.bit_width import *
from brevitas.core.quant import QuantType
from brevitas.core.stats import *
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE


class DefaultIntQuant(Injector):
    tensor_quant = RescalingIntQuant
    int_quant = IntQuant
    bit_width_impl = BitWidthConst
    bit_width = 8
    int_scaling_impl = FloatIntScaling
    affine_rescaling = False
    float_to_int_impl = RoundSte


class DefaultIntWeight(DefaultIntQuant):
    restrict_value_impl = LogFloatRestrictValue
    scaling_impl = ParameterStatsScaling
    stats_impl = AbsMax
    signed = True
    narrow_range = True


class DefaultWeightQuantInj(Injector):
    tensor_quant = None
    narrow_range = True


class DefaultIntAct(DefaultIntQuant):
    restrict_value_impl = FloatRestrictValue
    scaling_impl = ParameterScaling


def filter_kwargs(kwargs_prefix, kwargs: dict):
    return {k[len(kwargs_prefix):]: v for (k, v) in kwargs.items() if k.startswith(kwargs_prefix)}


def _check_name_value(qwi, name, value):
    return name in qwi and getattr(qwi, name) == value


def _solve_attr(qwi, value, solved_value, name: str, solved_key: str = None):
    if _check_name_value(qwi, name, value):
        if not isinstance(solved_value, dict):
            assert solved_key is not None
            qwi = qwi.let(**{solved_key: solved_value})
        else:
            qwi = qwi.let(**solved_value)
    return qwi


def _solve_weight_quant_type(qwi):
    solver = partial(_solve_attr, name='quant_type', solved_key='tensor_quant')
    qwi = solver(qwi, QuantType.FP, IdentityQuant)
    qwi = solver(qwi, QuantType.BINARY, BinaryQuant)
    qwi = solver(qwi, QuantType.TERNARY, TernaryQuant)
    qwi = solver(qwi, QuantType.INT, {'tensor_quant': RescalingIntQuant, 'int_quant': IntQuant})
    return qwi


def _solve_scaling_stats_op(qwi):
    solver = partial(_solve_attr, name='scaling_stats_op', solved_key='scaling_stats_impl')
    qwi = solver(qwi, StatsOp.MAX, AbsMax)
    qwi = solver(qwi, StatsOp.MAX_AVE, AbsMaxAve)
    qwi = solver(qwi, StatsOp.AVE, AbsAve)
    return qwi


def _solve_scaling_impl_type(qwi):
    solver = partial(_solve_attr, name='scaling_impl_type', solved_key='scaling_impl')
    qwi = solver(qwi, ScalingImplType.PARAMETER_FROM_STATS, ParameterScaling)
    qwi = solver(qwi, ScalingImplType.STATS, ParameterStatsScaling)
    qwi = solver(qwi, ScalingImplType.CONST, ConstScaling)
    qwi = solver(qwi, ScalingImplType.HE, ConstScaling)
    if 'scaling_override' in qwi: # TODO: deprecate
        qwi = solver(qwi, ScalingImplType.OVERRIDE, qwi.scaling_override)
    qwi = solver(qwi, ScalingImplType.AFFINE_STATS,
                 {'scaling_impl': ParameterStatsScaling, 'affine_rescaling': True})
    return qwi


def _solve_restrict_scaling_type(qwi):
    solver = partial(_solve_attr, name='restrict_scaling_type')
    qwi = solver(qwi, RestrictValueType.FP,
                 {'restrict_scaling_impl': FloatRestrictValue,
                  'int_scaling_impl': FloatIntScaling})
    qwi = solver(qwi, RestrictValueType.LOG_FP,
                 {'restrict_scaling_impl': LogFloatRestrictValue,
                  'int_scaling_impl': FloatIntScaling})
    qwi = solver(qwi, RestrictValueType.POWER_OF_TWO,
                 {'restrict_scaling_impl': PowerOfTwoRestrictValue,
                  'int_scaling_impl': PowerOfTwoIntScaling})
    return qwi


def _solve_bit_width_impl_type(qwi):
    solver = partial(_solve_attr, name='bit_width_impl_type', solved_key='bit_width_impl')
    if 'bit_width_impl_override' in qwi: #  TODO: deprecate
        return qwi.let(**{'bit_width_impl': qwi.bit_width_impl_override})
    qwi = solver(qwi, BitWidthImplType.CONST, BitWidthConst)
    qwi = solver(qwi, BitWidthImplType.PARAMETER, BitWidthParameter)
    return qwi


def _solve_float_to_int_impl(qwi, solver):
    qwi = solver(qwi, FloatToIntImplType.ROUND, RoundSte)
    qwi = solver(qwi, FloatToIntImplType.FLOOR, FloorSte)
    qwi = solver(qwi, FloatToIntImplType.CEIL, CeilSte)
    return qwi


def _solve_restrict_value_float_to_int_impl(qwi):
    impl = 'restrict_value_float_to_int_impl'
    impl_type = 'restrict_value_float_to_int_impl_type'
    solver = partial(_solve_attr, name=impl_type, solved_key=impl)
    if not impl in qwi and not impl_type in qwi:
        qwi = qwi.let(**{impl_type: FloatToIntImplType.CEIL})  # TODO: CEIL to ROUND
    qwi = _solve_float_to_int_impl(qwi, solver)
    return qwi


def _solve_tensor_quant_float_to_int_impl(qwi):
    impl = 'float_to_int_impl'
    impl_type = 'float_to_int_impl_type'
    solver = partial(_solve_attr, name=impl_type, solved_key=impl)
    if not impl in qwi and not impl_type in qwi:
        qwi = qwi.let(**{impl_type: FloatToIntImplType.ROUND})
    qwi = _solve_float_to_int_impl(qwi, solver)
    return qwi


def _solve_tensor_clamp_impl(qwi):
    impl = 'tensor_clamp_impl'
    already_set = impl in qwi
    if not already_set:
        if _check_name_value(qwi, 'bit_width_impl_type', BitWidthImplType.PARAMETER):
            qwi = qwi.let(**{impl: TensorClamp})
        elif _check_name_value(qwi, 'scaling_impl_type', ScalingImplType.PARAMETER_FROM_STATS):
            qwi = qwi.let(**{impl: TensorClamp})
        else:
            qwi = qwi.let(**{impl: TensorClampSte})
    return qwi


def _solve_scaling_shape(qwi, spoc):
    name = 'scaling_shape'
    if name not in qwi:
        if spoc: qwi = qwi.let(**{name: qwi.scaling_per_output_channel_shape})
        if not spoc: qwi = qwi.let(**{name: SCALING_SCALAR_SHAPE})
    return qwi


def _solve_scaling_stats_reduce_dim(qwi, spoc, ma):
    name = 'stats_reduce_dim'
    if name not in qwi:
        if spoc or ma: qwi = qwi.let(**{name: None})
        if not (spoc or ma): qwi = qwi.let(**{name: SCALING_STATS_REDUCE_DIM})
    return qwi


def _solve_scaling_stats_input_view_shape_impl(qwi, spoc, ma):
    name = 'scaling_stats_input_view_shape_impl'
    if name not in qwi:
        if spoc or ma: qwi = qwi.let(**{name: StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS})
        if not (spoc or ma): qwi = qwi.let(**{name: StatsInputViewShapeImpl.OVER_TENSOR})
    return qwi


def _solve_scaling_shapes_dims(qwi):
    spoc = _check_name_value(qwi, 'scaling_per_output_channel', True)
    ma = _check_name_value(qwi, 'scaling_stats_op', StatsOp.MAX_AVE)
    qwi = _solve_scaling_shape(qwi, spoc)
    qwi = _solve_scaling_stats_reduce_dim(qwi, spoc, ma)
    qwi = _solve_scaling_stats_input_view_shape_impl(qwi, spoc, ma)
    return qwi


def _solve_scaling_init_impl(qwi):
    #qwi
    pass


def _solve_enum_based_quant_weight_api(qwi):
    qwi = _solve_weight_quant_type(qwi)
    qwi = _solve_scaling_stats_op(qwi)
    qwi = _solve_scaling_impl_type(qwi)
    qwi = _solve_restrict_scaling_type(qwi)
    qwi = _solve_bit_width_impl_type(qwi)
    qwi = _solve_tensor_quant_float_to_int_impl(qwi)
    qwi = _solve_restrict_value_float_to_int_impl(qwi)
    qwi = _solve_tensor_clamp_impl(qwi)
    qwi = _solve_scaling_shapes_dims(qwi)
    return qwi


def _update_scaling_shape_data_from_weight_layer(qwi, weight_layer):
    per_channel_brodcast_shape = [1] * len(weight_layer.weight.size())
    per_channel_brodcast_shape[weight_layer.output_channel_dim] = weight_layer.out_channels
    qwi = qwi.let(scaling_per_output_channel_shape=tuple(per_channel_brodcast_shape))
    qwi = qwi.let(scaling_stats_input_concat_dim=weight_layer.output_channel_dim)
    qwi = qwi.let(returned_scale_shape=weight_layer.returned_scale_shape)
    return qwi


def update_weight_quant_inj(
        weight_layer: Module,
        quant_weight_inj: Injector,
        prefix: str,
        **kwargs):
    qwi = quant_weight_inj.let(**filter_kwargs(prefix, kwargs))
    qwi = _update_scaling_shape_data_from_weight_layer(qwi, weight_layer)
    qwi = _solve_enum_based_quant_weight_api(qwi)
    return qwi





