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

from typing import List
from functools import partial

from dependencies import this, value
from torch import nn

from brevitas.core.quant import *
from brevitas.core.function_wrapper import RoundSte, CeilSte, FloorSte, TensorClamp, TensorClampSte
from brevitas.core.function_wrapper import StatsInputViewShapeImpl
from brevitas.core.scaling import *
from brevitas.core.restrict_val import *
from brevitas.core.bit_width import *
from brevitas.core.quant import QuantType
from brevitas.core.stats import *
from brevitas.core.scaling import ScalingImplType, SCALAR_SHAPE
from brevitas.core.utils import StatelessBuffer
from brevitas.proxy.utils import ConvertRuntimeStatsToParameter
from . import BaseInjector as Injector


class EvaluateScalingInitImpl(Injector):

    @value
    def scaling_init(scaling_init_impl):
        scaling_init = scaling_init_impl()
        if isinstance(scaling_init, Tensor):
            return scaling_init.detach()
        else:
            return torch.tensor(scaling_init)


class ParameterFromStatsScalingInit:

    def __init__(self, parameter_stats_scaling: StatsFromParameterScaling):
        self.parameter_stats_scaling = parameter_stats_scaling
        self.ignored = StatelessBuffer(torch.tensor(0.0))

    def __call__(self):
        inp = self.ignored()
        return self.parameter_stats_scaling(inp)


class HeScalingInit:

    def __init__(self, tracked_parameter_list: List[torch.nn.Parameter]):
        self.tracked_parameter_list = tracked_parameter_list

    def __call__(self):
        scaling_init = 0.0
        # takes average of He scaling over parameter list
        for param in self.tracked_parameter_list:
            two_dim_param = param.view(param.shape[0], -1)
            scaling_init += math.sqrt(2.0 / two_dim_param.shape[1])
        scaling_init /= len(self.tracked_parameter_list)
        return torch.tensor(scaling_init)


class MinMaxScalingInit:

    def __init__(self, min_val: float, max_val: float):
        self.scaling_init = torch.tensor(max(abs(float(min_val)), abs(float(max_val))))

    def __call__(self):
        return self.scaling_init


def filter_kwargs(kwargs_prefix, kwargs: dict):
    return {k[len(kwargs_prefix):]: v for (k, v) in kwargs.items() if k.startswith(kwargs_prefix)}


def _check_name_value(qi, name, value):
    return name in qi and getattr(qi, name) == value


def _solve_attr(qi, value, solved_value, name: str, solved_key: str = None):
    if _check_name_value(qi, name, value):
        if not isinstance(solved_value, dict):
            assert solved_key is not None
            qi = qi.let(**{solved_key: solved_value})
        else:
            qi = qi.let(**solved_value)
    return qi


def _solve_bias_quant_type(qi):
    name = 'quant_type'
    solver = partial(_solve_attr, name=name)
    qi = solver(qi, QuantType.FP, {'tensor_quant': None})
    if _check_name_value(qi, name, QuantType.INT):
        qi = qi.let(**{'signed': True, 'narrow_range': False})
        if 'bit_width' in qi and 'scaling_impl' not in qi:
            qi = qi.let(**{'tensor_quant': PrescaledRestrictIntQuant,
                           'int_quant': IntQuant})
        elif 'bit_width' in qi and 'scaling_impl' in qi:
            qi = qi.let(**{'tensor_quant': RescalingIntQuant,
                           'int_quant': IntQuant})
        else:
            qi = qi.let(**{'tensor_quant': PrescaledRestrictIntQuantWithInputBitWidth,
                           'int_quant': IntQuant})
    return qi


def _solve_bit_width_impl_type(qi):
    solver = partial(_solve_attr, name='bit_width_impl_type', solved_key='bit_width_impl')
    qi = solver(qi, BitWidthImplType.CONST, BitWidthConst)
    qi = solver(qi, BitWidthImplType.PARAMETER, BitWidthParameter)
    return qi


def _solve_bias_bit_width_impl_type(qi):
    if 'bit_width' in qi:
        qi = qi.let(requires_input_bit_width=False)
    if 'bit_width' in qi and 'bit_width_impl_type' not in qi: # retrocomp. TODO deprecate
        qi = qi.let(bit_width_impl_type=BitWidthImplType.CONST)
    elif 'bit_width' not in qi and 'bit_width_impl_type' not in qi:
        qi = qi.let(**{'bit_width_impl': Identity, 'requires_input_bit_width': True})
    qi = _solve_bit_width_impl_type(qi)
    return qi


def _solve_bit_width_impl_override(qi):
    if 'bit_width_impl_override' in qi: #  TODO: deprecate
        return qi.let(**{'bit_width_impl': qi.bit_width_impl_override})
    return qi


def _solve_quant_type(qi, binary_quant_impl):
    solver = partial(_solve_attr, name='quant_type')
    qi = solver(qi, QuantType.FP, {'tensor_quant': None})
    qi = solver(qi, QuantType.BINARY,
                {'tensor_quant': binary_quant_impl, 'signed': True, 'narrow_range': False})
    qi = solver(qi, QuantType.TERNARY,
                {'tensor_quant': TernaryQuant, 'signed': True, 'narrow_range': True})
    qi = solver(qi, QuantType.INT, {'tensor_quant': RescalingIntQuant, 'int_quant': IntQuant})
    return qi


def _solve_weight_quant_type(qi):
    qi = _solve_quant_type(qi, binary_quant_impl=BinaryQuant)
    return qi


def _solve_act_quant_type(qi):
    qi = _solve_quant_type(qi, binary_quant_impl=ClampedBinaryQuant)
    return qi


def _solve_scaling_stats_op(qi):
    solver = partial(_solve_attr, name='scaling_stats_op', solved_key='scaling_stats_impl')
    qi = solver(qi, StatsOp.MAX, AbsMax)
    qi = solver(qi, StatsOp.MAX_AVE, AbsMaxAve)
    qi = solver(qi, StatsOp.AVE, AbsAve)
    qi = solver(qi, StatsOp.MEAN_SIGMA_STD, MeanSigmaStd)
    qi = solver(qi, StatsOp.MEAN_LEARN_SIGMA_STD, MeanLearnedSigmaStd)
    qi = solver(qi, StatsOp.PERCENTILE, AbsPercentile)
    if 'scaling_stats_sigma' in qi:
        qi = qi.let(sigma=this.scaling_stats_sigma)
    return qi


def _solve_scaling_override(qi):
    solver = partial(_solve_attr, name='scaling_impl_type', solved_key='scaling_impl')
    if 'scaling_override' in qi: # TODO: deprecate
        qi = solver(qi, ScalingImplType.OVERRIDE, qi.scaling_override)
    return qi


def _solve_weight_scaling_impl_type(qi):
    solver = partial(_solve_attr, name='scaling_impl_type', solved_key='scaling_impl')
    qi = solver(qi, ScalingImplType.PARAMETER, ParameterScaling)
    qi = solver(qi, ScalingImplType.PARAMETER_FROM_STATS, ParameterScaling)
    qi = solver(qi, ScalingImplType.CONST, ConstScaling)
    qi = solver(qi, ScalingImplType.HE, ConstScaling)
    qi = solver(qi, ScalingImplType.STATS,
                 {'scaling_impl': StatsFromParameterScaling, 'affine_rescaling': False})
    qi = solver(qi, ScalingImplType.AFFINE_STATS,
                 {'scaling_impl': StatsFromParameterScaling, 'affine_rescaling': True})
    return qi


def _solve_act_scaling_impl_type(qi):
    solver = partial(_solve_attr, name='scaling_impl_type', solved_key='scaling_impl')
    qi = solver(qi, ScalingImplType.PARAMETER, ParameterScaling)
    qi = solver(qi, ScalingImplType.CONST, ConstScaling)
    qi = solver(qi, ScalingImplType.PARAMETER_FROM_STATS, ParameterFromRuntimeStatsScaling)
    qi = solver(qi, ScalingImplType.STATS,
                 {'scaling_impl': RuntimeStatsScaling, 'affine_rescaling': False})
    qi = solver(qi, ScalingImplType.AFFINE_STATS,
                 {'scaling_impl': RuntimeStatsScaling, 'affine_rescaling': True})
    return qi


def _solve_restrict_scaling_type(qi):
    solver = partial(_solve_attr, name='restrict_scaling_type')
    qi = solver(qi, RestrictValueType.FP,
                {'restrict_scaling_impl': FloatRestrictValue,
                  'int_scaling_impl': IntScaling})
    qi = solver(qi, RestrictValueType.LOG_FP,
                {'restrict_scaling_impl': LogFloatRestrictValue,
                  'int_scaling_impl': IntScaling})
    qi = solver(qi, RestrictValueType.POWER_OF_TWO,
                 {'restrict_scaling_impl': PowerOfTwoRestrictValue,
                  'int_scaling_impl': PowerOfTwoIntScaling})
    return qi


def _solve_restrict_bit_width_type(qi):
    solver = partial(
        _solve_attr, name='restrict_bit_width_type', solved_key='restrict_bit_width_impl')
    qi = solver(qi, RestrictValueType.FP, FloatRestrictValue)
    qi = solver(qi, RestrictValueType.LOG_FP, LogFloatRestrictValue)
    qi = solver(qi, RestrictValueType.INT, IntRestrictValue)
    qi = solver(qi, RestrictValueType.POWER_OF_TWO, PowerOfTwoRestrictValue)
    return qi


def _solve_float_to_int_impl(qi, solver):
    qi = solver(qi, FloatToIntImplType.ROUND, RoundSte)
    qi = solver(qi, FloatToIntImplType.FLOOR, FloorSte)
    qi = solver(qi, FloatToIntImplType.CEIL, CeilSte)
    return qi


def _solve_restrict_value_float_to_int_impl(qi):
    impl = 'restrict_value_float_to_int_impl'
    impl_type = 'restrict_value_float_to_int_impl_type'
    solver = partial(_solve_attr, name=impl_type, solved_key=impl)
    if not impl in qi and not impl_type in qi:
        qi = qi.let(**{impl_type: FloatToIntImplType.ROUND})
    qi = _solve_float_to_int_impl(qi, solver)
    return qi


def _solve_tensor_quant_float_to_int_impl(qi):
    impl = 'float_to_int_impl'
    impl_type = 'float_to_int_impl_type'
    solver = partial(_solve_attr, name=impl_type, solved_key=impl)
    if not impl in qi and not impl_type in qi:
        qi = qi.let(**{impl_type: FloatToIntImplType.ROUND}) # for retrocomp, TODO deprecate
    qi = _solve_float_to_int_impl(qi, solver)
    return qi


def _solve_weight_tensor_clamp_impl(qi):
    impl = 'tensor_clamp_impl'
    already_set = impl in qi
    if not already_set:
        if _check_name_value(qi, 'bit_width_impl_type', BitWidthImplType.PARAMETER):
            qi = qi.let(**{impl: TensorClamp})
        elif _check_name_value(qi, 'scaling_impl_type', ScalingImplType.PARAMETER_FROM_STATS):
            qi = qi.let(**{impl: TensorClamp})
        elif _check_name_value(qi, 'scaling_impl_type', ScalingImplType.PARAMETER):
            qi = qi.let(**{impl: TensorClamp})
        else:
            qi = qi.let(**{impl: TensorClampSte})
    return qi


def _solve_act_tensor_clamp_impl(qi):
    impl = 'tensor_clamp_impl'
    already_set = impl in qi
    if not already_set:
            qi = qi.let(**{impl: TensorClamp})
    return qi


def _solve_scaling_shape(qi, spoc, per_channel_shape_attr):
    name = 'scaling_shape'
    if name not in qi:
        if spoc: qi = qi.let(**{name: per_channel_shape_attr})
        if not spoc: qi = qi.let(**{name: SCALAR_SHAPE})
    return qi


def _solve_scaling_stats_reduce_dim(qi, spoc, ma):
    name = 'stats_reduce_dim'
    if name not in qi:
        if spoc or ma: qi = qi.let(**{name: SCALING_STATS_REDUCE_DIM})
        else: qi = qi.let(**{name: None})
    return qi


def _solve_scaling_stats_input_view_shape_impl(qi, spoc, ma):
    name = 'scaling_stats_input_view_shape_impl'
    if name not in qi:
        if spoc or ma: qi = qi.let(**{name: StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS})
        else: qi = qi.let(**{name: StatsInputViewShapeImpl.OVER_TENSOR})
    return qi


def _solve_scaling_shapes_dims(qi, per_channel_shape_attr):
    if 'scaling_per_channel' in qi: # act
        qi = qi.let(scaling_per_output_channel=this.scaling_per_channel)
    spoc = _check_name_value(qi, 'scaling_per_output_channel', True)
    ma = _check_name_value(qi, 'scaling_stats_op', StatsOp.MAX_AVE)
    qi = _solve_scaling_shape(qi, spoc, per_channel_shape_attr)
    qi = _solve_scaling_stats_reduce_dim(qi, spoc, ma)
    qi = _solve_scaling_stats_input_view_shape_impl(qi, spoc, ma)
    return qi


def _solve_weight_scaling_shapes_dims(qi):
    qi = _solve_scaling_shapes_dims(qi, this.scaling_per_output_channel_shape)
    return qi


def _solve_act_scaling_shapes_dims(qi):
    qi = _solve_scaling_shapes_dims(qi, this.per_channel_broadcastable_shape)
    return qi


def _solve_weight_scaling_init_impl(qi):
    name = 'scaling_impl_type'
    if _check_name_value(qi, name, ScalingImplType.CONST):
        qi = qi.let(scaling_init=this.scaling_const)
    if _check_name_value(qi, name, ScalingImplType.PARAMETER):
        qi = qi.let(scaling_init=this.scaling_const)
    if _check_name_value(qi, name, ScalingImplType.PARAMETER_FROM_STATS):
        qi = qi & EvaluateScalingInitImpl
        qi = qi.let(scaling_init_impl=ParameterFromStatsScalingInit)
        qi = qi.let(parameter_stats_scaling=StatsFromParameterScaling)
    elif _check_name_value(qi, name, ScalingImplType.HE):
        qi = qi & EvaluateScalingInitImpl
        qi = qi.let(scaling_init_impl=HeScalingInit)
    return qi


def _solve_act_scaling_init_impl(qi):
    name = 'scaling_impl_type'
    qi = qi & EvaluateScalingInitImpl
    p = _check_name_value(qi, name, ScalingImplType.PARAMETER)
    c = _check_name_value(qi, name, ScalingImplType.CONST)
    signed = _check_name_value(qi, 'signed', True)
    if not signed:
        qi = qi.let(min_val=0)
    if p or c:
        qi = qi.let(scaling_init_impl=MinMaxScalingInit)
    return qi


def _solve_act_scaling_conversion(qi):
    if _check_name_value(qi, 'scaling_impl_type', ScalingImplType.PARAMETER):
        qi = qi.let(update_state_dict_impl=ConvertRuntimeStatsToParameter)
    return qi


def _solve_enum_based_quant_weight_api(qi):
    qi = _solve_weight_quant_type(qi)
    qi = _solve_scaling_stats_op(qi)
    qi = _solve_weight_scaling_impl_type(qi)
    qi = _solve_restrict_scaling_type(qi)
    qi = _solve_bit_width_impl_type(qi)
    qi = _solve_restrict_bit_width_type(qi)
    qi = _solve_bit_width_impl_override(qi)
    qi = _solve_scaling_override(qi)
    qi = _solve_tensor_quant_float_to_int_impl(qi)
    qi = _solve_restrict_value_float_to_int_impl(qi)
    qi = _solve_weight_tensor_clamp_impl(qi)
    qi = _solve_weight_scaling_shapes_dims(qi)
    qi = _solve_weight_scaling_init_impl(qi)
    return qi


def _solve_enum_based_quant_bias_api(qi):
    qi = _solve_scaling_stats_op(qi)
    qi = _solve_weight_scaling_impl_type(qi)
    qi = _solve_restrict_scaling_type(qi)
    qi = _solve_weight_scaling_shapes_dims(qi)
    qi = _solve_weight_scaling_init_impl(qi)
    qi = _solve_restrict_value_float_to_int_impl(qi)
    qi = qi.let(requires_input_scale='scaling_impl' not in qi)
    qi = _solve_bias_quant_type(qi)
    qi = _solve_bias_bit_width_impl_type(qi)
    qi = _solve_tensor_quant_float_to_int_impl(qi)
    if 'tensor_clamp_impl' not in qi:
        qi = qi.let(tensor_clamp_impl=TensorClamp)
    return qi


def _solve_enum_based_quant_act_api(qi):
    qi = _solve_act_quant_type(qi)
    qi = _solve_bit_width_impl_override(qi)
    qi = _solve_bit_width_impl_type(qi)
    qi = _solve_restrict_bit_width_type(qi)
    qi = _solve_tensor_quant_float_to_int_impl(qi)
    qi = _solve_scaling_stats_op(qi)
    qi = _solve_scaling_override(qi)
    qi = _solve_restrict_scaling_type(qi)
    qi = _solve_restrict_value_float_to_int_impl(qi)
    qi = _solve_act_tensor_clamp_impl(qi)
    qi = _solve_act_scaling_impl_type(qi)
    qi = _solve_act_scaling_shapes_dims(qi)
    qi = _solve_act_scaling_init_impl(qi)
    qi = _solve_act_scaling_conversion(qi)
    return qi


def _update_act_impl(qi):
    # retrocompatibility TODO deprecate
    min_val_set = 'min_val' in qi
    max_val_set = 'max_val' in qi
    signed_set = 'signed' in qi
    quant_type_fp = _check_name_value(qi, 'quant_type', QuantType.FP)
    unsigned_attrs = {'min_val': 0.0, 'signed': False}
    if isinstance(qi.act_impl, nn.ReLU) and not min_val_set and not signed_set:
        qi = qi.let(**unsigned_attrs)
    elif isinstance(qi.act_impl, nn.Sigmoid) and not min_val_set and not max_val_set and not signed_set:
        qi = qi.let(max_val_set=1.0, **unsigned_attrs)
    elif isinstance(qi.act_impl, nn.Tanh) and not min_val_set and not signed_set:
        qi = qi.let({'signed': True, 'min_val': -1.0, 'max_val': 1.0})
    elif isinstance(qi.act_impl, nn.Hardtanh) and not quant_type_fp:
        qi = qi.let(act_impl=None)
    return qi


def update_act_quant_injector(
        act_layer: Module,
        act_quant_injector: Injector,
        prefix: str,
        **kwargs):
    qi = act_quant_injector.let(**filter_kwargs(prefix, kwargs))
    qi = _update_act_impl(qi)
    qi = _solve_enum_based_quant_act_api(qi)
    return qi


def _update_from_weight_layer(qi, weight_layer):
    per_channel_brodcast_shape = [1] * len(weight_layer.weight.size())
    per_channel_brodcast_shape[weight_layer.output_channel_dim] = weight_layer.out_channels
    qi = qi.let(scaling_per_output_channel_shape=tuple(per_channel_brodcast_shape))
    qi = qi.let(scaling_stats_input_concat_dim=weight_layer.output_channel_dim)
    return qi


def _update_from_bias_layer(qi, bias_layer):
    per_channel_brodcast_shape = (bias_layer.out_channels,)
    qi = qi.let(scaling_per_output_channel_shape=tuple(per_channel_brodcast_shape))
    qi = qi.let(scaling_stats_input_concat_dim=bias_layer.output_channel_dim)
    return qi


def update_weight_quant_injector(
        weight_layer: Module,
        weight_quant_injector: Injector,
        prefix: str,
        **kwargs):
    qi = weight_quant_injector.let(**filter_kwargs(prefix, kwargs))
    qi = _update_from_weight_layer(qi, weight_layer)
    qi = _solve_enum_based_quant_weight_api(qi)
    return qi


def update_bias_quant_injector(
        bias_layer: Module,
        bias_quant_injector: Injector,
        prefix: str,
        **kwargs):
    qi = bias_quant_injector.let(**filter_kwargs(prefix, kwargs))
    qi = _update_from_bias_layer(qi, bias_layer)
    qi = _solve_enum_based_quant_bias_api(qi)
    return qi


def _solve_bit_width_to_remove_impl(qi, name):
    key = 'bit_width_to_remove_impl'
    solver = partial(_solve_attr, name=name, solved_key=key)
    qi = solver(qi, BitWidthImplType.CONST, BitWidthConst)
    qi = solver(qi, BitWidthImplType.PARAMETER, RemoveBitwidthParameter)
    return qi


def _solve_trunc_quant_type(qi):
    solver = partial(_solve_attr, name='quant_type', solved_key='tensor_quant')
    qi = solver(qi, QuantType.FP, None)
    qi = solver(qi, QuantType.INT, TruncIntQuant)
    return qi


def _solve_trunc_float_to_int_impl_type(qi):
    impl = 'float_to_int_impl'
    impl_type = 'float_to_int_impl_type'
    solver = partial(_solve_attr, name=impl_type, solved_key=impl)
    qi = _solve_float_to_int_impl(qi, solver)
    return qi


def _solve_enum_based_quant_trunc_api(qi):
    qi = _solve_trunc_quant_type(qi)
    qi = _solve_bit_width_impl_type(qi)
    qi = _solve_trunc_float_to_int_impl_type(qi)
    return qi


def update_trunc_quant_injector(
        trunc_layer: Module,
        trunc_quant_injector: Injector,
        prefix: str,
        **kwargs):
    qi = trunc_quant_injector.let(**filter_kwargs(prefix, kwargs))
    qi = _solve_enum_based_quant_trunc_api(qi)
    return qi


def _solve_clamp_quant_type(qi):
    solver = partial(_solve_attr, name='quant_type')
    qi = solver(qi, QuantType.FP, {'tensor_quant': None, 'msb_clamp_bit_width_impl': None})
    qi = solver(qi, QuantType.INT,
                {'tensor_quant': PrescaledRestrictIntQuantWithInputBitWidth,
                 'bit_width_impl': MsbClampBitWidth})
    return qi


def _solve_msb_clamp_bit_width_impl_type(qi):
    name = 'msb_clamp_bit_width_impl_type'
    qi = _solve_bit_width_to_remove_impl(qi, name)
    return qi


def _solve_enum_based_quant_clamp_api(qi):
    qi = _solve_clamp_quant_type(qi)
    qi = _solve_msb_clamp_bit_width_impl_type(qi)
    if 'tensor_clamp_impl' not in qi:
        qi = qi.let(tensor_clamp_impl=TensorClamp)
    if 'float_to_int_impl' not in qi:  # this really shouldn't be up for change
        qi = qi.let(float_to_int_impl=RoundSte)
    return qi


def update_clamp_quant_injector(
        clamp_layer: Module,
        clamp_quant_injector: Injector,
        prefix: str,
        **kwargs):
    qi = clamp_quant_injector.let(**filter_kwargs(prefix, kwargs))
    qi = _solve_enum_based_quant_clamp_api(qi)
    return qi


