from hypothesis import given
from hypothesis import strategies as st
from common import *
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE, StatsInputViewShapeImpl, ParameterStatsScaling
from brevitas.core.scaling import StandaloneScaling, RuntimeStatsScaling, AffineRescaling
from brevitas.core.stats import AbsAve, AbsMax, AbsMaxAve
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.bit_width import BitWidthImplType, BitWidthConst, BitWidthParameter
from brevitas.core.quant import QuantType, PrescaledRestrictIntQuant, PrescaledRestrictIntQuantWithInputBitWidth
from brevitas.core.restrict_val import RestrictValueType
import pytest
from brevitas.core.stats import StatsOp
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy
from brevitas.proxy.runtime_quant import ActivationQuantProxy
import torch
from torch.nn import Parameter
from brevitas.core.function_wrapper import TensorClamp, TensorClampSte, Identity

# Constants
OUTCHANNEL = 20
SCALING_MIN_VAL = 0.0
MIN_BIT_WIDTH = 3
MAX_BIT_WIDTH = 8
SHAPE = (1, 10, 5, 5)

# Situational Constants
# All following values are not important for the particular tests of this module. Their effect on the
# instantiated objects is evaluated separately, in the individual tests for each one of the modules.
NARROW_RANGE = True
RESTRICT_TYPE = RestrictValueType.LOG_FP
SCALING_CONST = 1.0
SCALING_SHAPE = SCALING_SCALAR_SHAPE
STATS_INPUT_VIEW_SHAPE_IMPL = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
SCALING_STATS_INPUT_CONCAT_DIM = 0
BIT_WIDTH_IMPL_OVERRIDE = None
TERNARY_THRESHOLD = 0.5
SCALING_STATS_SIGMA = 3.0
SCALING_OVERRIDE = None
OVERRIDE_PRETRAINED_BITWIDTH = False
RESTRICT_BIT_WIDTH_TYPE = RestrictValueType.INT
SCALING_STATS_REDUCE_DIM = None

# Runtime Proxy Constants
SIGNED = True
MAX_VAL = 10
MIN_VAL = -10
FLOAT_TO_INT_IMPL_TYPE = FloatToIntImplType.ROUND
SCALING_STATS_BUFFER_MOMENTUM = 0.1
SCALING_STATS_PERMUTE_DIMS = (1, 0, 2, 3)
PER_CHANNEL_BROADCASTABLE_SHAPE = None
SCALING_PER_CHANNEL = False
RESTRICT_SCALING_TYPE = RestrictValueType.FP

# Hypothesis strategies
bit_width_st = st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH)

# Weight Quant Proxy parametrize options
scaling_impl_type_options = [(ScalingImplType.STATS), (ScalingImplType.PARAMETER_FROM_STATS),
                             (ScalingImplType.AFFINE_STATS)]
scaling_impl_const_options = [(ScalingImplType.CONST), (ScalingImplType.HE)]
scaling_stats_op_options = [(StatsOp.MAX, AbsMax), (StatsOp.AVE, AbsAve), (StatsOp.MAX_AVE, AbsMaxAve)]
bit_width_impl_options = [(BitWidthImplType.CONST, BitWidthConst), (BitWidthImplType.PARAMETER, BitWidthParameter)]
quant_type_options = [(QuantType.BINARY, 1), (QuantType.TERNARY, 2)]

# Runtime Proxy parametrize options
runtime_scaling_impl_type_options = [(ScalingImplType.CONST), (ScalingImplType.PARAMETER)]
runtime_scaling_impl_type_stats_options = [(ScalingImplType.STATS), (ScalingImplType.AFFINE_STATS)]
runtime_quant_type_options = [(QuantType.BINARY), (QuantType.INT)]


@pytest.mark.parametrize('scaling_impl_type', scaling_impl_type_options)
@pytest.mark.parametrize('scaling_stats_op, scaling_impl_op', scaling_stats_op_options)
@given(bit_width=bit_width_st)
def test_weight_quant_proxy_scaling_nobt(bit_width, scaling_impl_type, scaling_stats_op, scaling_impl_op):
    # Const for this suite of tests
    bit_width_impl = BitWidthImplType.CONST
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    quant_type = QuantType.INT
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    reduce_dim = None if scaling_stats_op is not StatsOp.MAX_AVE else 0

    obj = WeightQuantProxy(tracked_parameter_list_init=inp,
                           bit_width=bit_width,
                           quant_type=quant_type,
                           narrow_range=NARROW_RANGE,
                           restrict_scaling_type=RESTRICT_TYPE,
                           scaling_impl_type=scaling_impl_type,
                           scaling_stats_op=scaling_stats_op,
                           scaling_shape=SCALING_SHAPE,
                           bit_width_impl_type=bit_width_impl,
                           scaling_stats_input_concat_dim=SCALING_STATS_INPUT_CONCAT_DIM,
                           scaling_stats_input_view_shape_impl=STATS_INPUT_VIEW_SHAPE_IMPL,
                           bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                           max_overall_bit_width=MAX_BIT_WIDTH,
                           min_overall_bit_width=MIN_BIT_WIDTH,
                           override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                           restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                           scaling_const=SCALING_CONST,
                           scaling_min_val=SCALING_MIN_VAL,
                           scaling_override=SCALING_OVERRIDE,
                           scaling_stats_reduce_dim=reduce_dim,
                           scaling_stats_sigma=SCALING_STATS_SIGMA,
                           ternary_threshold=TERNARY_THRESHOLD)

    if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
        assert isinstance(obj.tensor_quant.scaling_impl, StandaloneScaling)
        assert (obj.tensor_quant.scaling_impl.const_value is None) and \
               (obj.tensor_quant.scaling_impl.learned_value.requires_grad==True)
    else:
        assert isinstance(obj.tensor_quant.scaling_impl, ParameterStatsScaling)
        actual_stats_implementation = obj.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl
        assert isinstance(actual_stats_implementation, scaling_impl_op)

    tensor_clamp_impl = obj.tensor_quant.int_quant.tensor_clamp_impl
    expected_tensor_clamp_impl = TensorClamp if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS else TensorClampSte
    assert isinstance(tensor_clamp_impl, expected_tensor_clamp_impl)


@pytest.mark.parametrize('scaling_impl_type', scaling_impl_type_options)
@pytest.mark.parametrize('scaling_stats_op, scaling_impl_op', scaling_stats_op_options)
@pytest.mark.parametrize('quant_type, bit_width', quant_type_options)
def test_weight_quant_proxy_scaling_bt(scaling_impl_type, scaling_stats_op, scaling_impl_op, quant_type, bit_width):
    # Const for this suite of tests
    if scaling_impl_type is ScalingImplType.PARAMETER_FROM_STATS:
        pytest.xfail("Not supported")
    bit_width_impl = BitWidthImplType.CONST
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    bit_width = torch.tensor(bit_width, dtype=torch.float)

    reduce_dim = None if scaling_stats_op is not StatsOp.MAX_AVE else 0

    obj = WeightQuantProxy(tracked_parameter_list_init=inp,
                           bit_width=bit_width,
                           quant_type=quant_type,
                           narrow_range=NARROW_RANGE,
                           restrict_scaling_type=RESTRICT_TYPE,
                           scaling_impl_type=scaling_impl_type,
                           scaling_stats_op=scaling_stats_op,
                           scaling_shape=SCALING_SHAPE,
                           bit_width_impl_type=bit_width_impl,
                           scaling_stats_input_concat_dim=SCALING_STATS_INPUT_CONCAT_DIM,
                           scaling_stats_input_view_shape_impl=STATS_INPUT_VIEW_SHAPE_IMPL,
                           bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                           max_overall_bit_width=MAX_BIT_WIDTH,
                           min_overall_bit_width=MIN_BIT_WIDTH,
                           override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                           restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                           scaling_const=SCALING_CONST,
                           scaling_min_val=SCALING_MIN_VAL,
                           scaling_override=SCALING_OVERRIDE,
                           scaling_stats_reduce_dim=reduce_dim,
                           scaling_stats_sigma=SCALING_STATS_SIGMA,
                           ternary_threshold=TERNARY_THRESHOLD)

    assert isinstance(obj.tensor_quant.scaling_impl, ParameterStatsScaling)
    actual_implementation = obj.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl
    assert isinstance(actual_implementation, scaling_impl_op)


@pytest.mark.parametrize('scaling_impl_type', scaling_impl_const_options)
@given(bit_width=bit_width_st)
def test_weight_quant_proxy_scaling_const_nobt(bit_width, scaling_impl_type):
    # Const for this suite of tests
    bit_width_impl = BitWidthImplType.CONST
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    quant_type = QuantType.INT
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    obj = WeightQuantProxy(tracked_parameter_list_init=inp,
                           bit_width=bit_width,
                           quant_type=quant_type,
                           narrow_range=NARROW_RANGE,
                           restrict_scaling_type=RESTRICT_TYPE,
                           scaling_impl_type=scaling_impl_type,
                           scaling_stats_op=StatsOp.MAX,
                           scaling_shape=SCALING_SHAPE,
                           bit_width_impl_type=bit_width_impl,
                           scaling_stats_input_concat_dim=SCALING_STATS_INPUT_CONCAT_DIM,
                           scaling_stats_input_view_shape_impl=STATS_INPUT_VIEW_SHAPE_IMPL,
                           bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                           max_overall_bit_width=MAX_BIT_WIDTH,
                           min_overall_bit_width=MIN_BIT_WIDTH,
                           override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                           restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                           scaling_const=SCALING_CONST,
                           scaling_min_val=SCALING_MIN_VAL,
                           scaling_override=SCALING_OVERRIDE,
                           scaling_stats_reduce_dim=SCALING_STATS_REDUCE_DIM,
                           scaling_stats_sigma=SCALING_STATS_SIGMA,
                           ternary_threshold=TERNARY_THRESHOLD)

    assert isinstance(obj.tensor_quant.scaling_impl, StandaloneScaling)
    assert (obj.tensor_quant.scaling_impl.const_value is not None) and \
           (obj.tensor_quant.scaling_impl.learned_value is None)


@pytest.mark.parametrize('scaling_impl_type', scaling_impl_const_options)
@pytest.mark.parametrize('quant_type, bit_width', quant_type_options)
def test_weight_quant_proxy_scaling_const_bt(quant_type, bit_width, scaling_impl_type):
    # Const for this suite of tests
    bit_width_impl = BitWidthImplType.CONST
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    obj = WeightQuantProxy(tracked_parameter_list_init=inp,
                           bit_width=bit_width,
                           quant_type=quant_type,
                           narrow_range=NARROW_RANGE,
                           restrict_scaling_type=RESTRICT_TYPE,
                           scaling_impl_type=scaling_impl_type,
                           scaling_stats_op=StatsOp.MAX,
                           scaling_shape=SCALING_SHAPE,
                           bit_width_impl_type=bit_width_impl,
                           scaling_stats_input_concat_dim=SCALING_STATS_INPUT_CONCAT_DIM,
                           scaling_stats_input_view_shape_impl=STATS_INPUT_VIEW_SHAPE_IMPL,
                           bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                           max_overall_bit_width=MAX_BIT_WIDTH,
                           min_overall_bit_width=MIN_BIT_WIDTH,
                           override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                           restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                           scaling_const=SCALING_CONST,
                           scaling_min_val=SCALING_MIN_VAL,
                           scaling_override=SCALING_OVERRIDE,
                           scaling_stats_reduce_dim=SCALING_STATS_REDUCE_DIM,
                           scaling_stats_sigma=SCALING_STATS_SIGMA,
                           ternary_threshold=TERNARY_THRESHOLD)

    assert isinstance(obj.tensor_quant.scaling_impl, StandaloneScaling)
    assert (obj.tensor_quant.scaling_impl.const_value is not None) and \
           (obj.tensor_quant.scaling_impl.learned_value is None)


@pytest.mark.parametrize('bit_width_impl, expected_impl', bit_width_impl_options)
@given(bit_width=bit_width_st)
def test_weight_quant_proxy_bit_width_impl(bit_width_impl, bit_width, expected_impl):
    # Const for this suite of tests
    scaling_impl_type = ScalingImplType.CONST
    quant_type = QuantType.INT
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    obj = WeightQuantProxy(tracked_parameter_list_init=inp,
                           bit_width=bit_width,
                           quant_type=quant_type,
                           narrow_range=NARROW_RANGE,
                           restrict_scaling_type=RESTRICT_TYPE,
                           scaling_impl_type=scaling_impl_type,
                           scaling_stats_op=StatsOp.MAX,
                           scaling_shape=SCALING_SHAPE,
                           bit_width_impl_type=bit_width_impl,
                           scaling_stats_input_concat_dim=SCALING_STATS_INPUT_CONCAT_DIM,
                           scaling_stats_input_view_shape_impl=STATS_INPUT_VIEW_SHAPE_IMPL,
                           bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                           max_overall_bit_width=MAX_BIT_WIDTH,
                           min_overall_bit_width=MIN_BIT_WIDTH,
                           override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                           restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                           scaling_const=SCALING_CONST,
                           scaling_min_val=SCALING_MIN_VAL,
                           scaling_override=SCALING_OVERRIDE,
                           scaling_stats_reduce_dim=SCALING_STATS_REDUCE_DIM,
                           scaling_stats_sigma=SCALING_STATS_SIGMA,
                           ternary_threshold=TERNARY_THRESHOLD)

    assert isinstance(obj.tensor_quant.msb_clamp_bit_width_impl, expected_impl)


@pytest.mark.parametrize('scaling_impl_type', runtime_scaling_impl_type_options)
@pytest.mark.parametrize('quant_type', runtime_quant_type_options)
@given(bit_width=bit_width_st)
def test_runtime_proxy_stats(scaling_impl_type, bit_width, quant_type):
    bit_width_impl = BitWidthImplType.CONST
    scaling_stats_op = StatsOp.MAX
    if quant_type is QuantType.BINARY:
        bit_width = 1
    bit_width = torch.tensor(bit_width, dtype=torch.float)

    obj = ActivationQuantProxy(activation_impl=torch.nn.Identity(),
                               bit_width=bit_width,
                               signed=SIGNED,
                               narrow_range=NARROW_RANGE,
                               min_val=MIN_VAL,
                               max_val=MAX_VAL,
                               quant_type=quant_type,
                               float_to_int_impl_type=FLOAT_TO_INT_IMPL_TYPE,
                               bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                               bit_width_impl_type=bit_width_impl,
                               max_overall_bit_width=MAX_BIT_WIDTH,
                               min_overall_bit_width=MIN_BIT_WIDTH,
                               override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                               per_channel_broadcastable_shape=PER_CHANNEL_BROADCASTABLE_SHAPE,
                               restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                               restrict_scaling_type=RESTRICT_SCALING_TYPE,
                               scaling_impl_type=scaling_impl_type,
                               scaling_min_val=SCALING_MIN_VAL,
                               scaling_override=SCALING_OVERRIDE,
                               scaling_per_channel=SCALING_PER_CHANNEL,
                               scaling_stats_buffer_momentum=SCALING_STATS_BUFFER_MOMENTUM,
                               scaling_stats_op=scaling_stats_op,
                               scaling_stats_permute_dims=SCALING_STATS_PERMUTE_DIMS,
                               scaling_stats_sigma=SCALING_STATS_SIGMA)

    assert isinstance(obj.fused_activation_quant_proxy.tensor_quant.scaling_impl, StandaloneScaling)
    if scaling_impl_type == ScalingImplType.PARAMETER:
        assert (obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.const_value is None) and \
                   (obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.learned_value.requires_grad==True)
    else:
        assert (obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.const_value is not None) and \
                   (obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.learned_value is None)


@pytest.mark.parametrize('scaling_impl_type', runtime_scaling_impl_type_stats_options)
@pytest.mark.parametrize('scaling_stats_op, scaling_impl_op', scaling_stats_op_options)
@pytest.mark.parametrize('quant_type', runtime_quant_type_options)
@given(bit_width=bit_width_st)
def test_runtime_proxy_stats(scaling_stats_op, scaling_impl_op, scaling_impl_type, bit_width, quant_type):
    bit_width_impl = BitWidthImplType.CONST
    if quant_type is QuantType.BINARY:
        bit_width = 1
    bit_width = torch.tensor(bit_width, dtype=torch.float)

    obj = ActivationQuantProxy(activation_impl=torch.nn.Identity(),
                               bit_width=bit_width,
                               signed=SIGNED,
                               narrow_range=NARROW_RANGE,
                               min_val=MIN_VAL,
                               max_val=MAX_VAL,
                               quant_type=quant_type,
                               float_to_int_impl_type=FLOAT_TO_INT_IMPL_TYPE,
                               bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                               bit_width_impl_type=bit_width_impl,
                               max_overall_bit_width=MAX_BIT_WIDTH,
                               min_overall_bit_width=MIN_BIT_WIDTH,
                               override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                               per_channel_broadcastable_shape=PER_CHANNEL_BROADCASTABLE_SHAPE,
                               restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                               restrict_scaling_type=RESTRICT_SCALING_TYPE,
                               scaling_impl_type=scaling_impl_type,
                               scaling_min_val=SCALING_MIN_VAL,
                               scaling_override=SCALING_OVERRIDE,
                               scaling_per_channel=SCALING_PER_CHANNEL,
                               scaling_stats_buffer_momentum=SCALING_STATS_BUFFER_MOMENTUM,
                               scaling_stats_op=scaling_stats_op,
                               scaling_stats_permute_dims=SCALING_STATS_PERMUTE_DIMS,
                               scaling_stats_sigma=SCALING_STATS_SIGMA)

    assert isinstance(obj.fused_activation_quant_proxy.tensor_quant.scaling_impl, RuntimeStatsScaling)
    actual_implementation = obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.runtime_stats.stats.stats_impl
    assert isinstance(actual_implementation, scaling_impl_op)
    affine_impl = obj.fused_activation_quant_proxy.tensor_quant.scaling_impl.stats_scaling_impl.affine_rescaling
    expected_impl = AffineRescaling if scaling_impl_type is ScalingImplType.AFFINE_STATS else Identity
    assert isinstance(affine_impl, expected_impl)


@pytest.mark.parametrize('bit_width_impl, expected_impl', bit_width_impl_options)
@given(bit_width=bit_width_st)
def test_runtime_proxy_stats(bit_width_impl, expected_impl, bit_width):
    scaling_impl_type = ScalingImplType.CONST
    scaling_stats_op = StatsOp.MAX
    quant_type = QuantType.INT
    bit_width = torch.tensor(bit_width, dtype=torch.float)

    obj = ActivationQuantProxy(activation_impl=torch.nn.Identity(),
                               bit_width=bit_width,
                               signed=SIGNED,
                               narrow_range=NARROW_RANGE,
                               min_val=MIN_VAL,
                               max_val=MAX_VAL,
                               quant_type=quant_type,
                               float_to_int_impl_type=FLOAT_TO_INT_IMPL_TYPE,
                               bit_width_impl_override=BIT_WIDTH_IMPL_OVERRIDE,
                               bit_width_impl_type=bit_width_impl,
                               max_overall_bit_width=MAX_BIT_WIDTH,
                               min_overall_bit_width=MIN_BIT_WIDTH,
                               override_pretrained_bit_width=OVERRIDE_PRETRAINED_BITWIDTH,
                               per_channel_broadcastable_shape=PER_CHANNEL_BROADCASTABLE_SHAPE,
                               restrict_bit_width_type=RESTRICT_BIT_WIDTH_TYPE,
                               restrict_scaling_type=RESTRICT_SCALING_TYPE,
                               scaling_impl_type=scaling_impl_type,
                               scaling_min_val=SCALING_MIN_VAL,
                               scaling_override=SCALING_OVERRIDE,
                               scaling_per_channel=SCALING_PER_CHANNEL,
                               scaling_stats_buffer_momentum=SCALING_STATS_BUFFER_MOMENTUM,
                               scaling_stats_op=scaling_stats_op,
                               scaling_stats_permute_dims=SCALING_STATS_PERMUTE_DIMS,
                               scaling_stats_sigma=SCALING_STATS_SIGMA)

    assert isinstance(obj.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl, expected_impl)


@given(bit_width=bit_width_st)
def test_bias_quant_proxy_external_bitwidth(bit_width):
    quant_type = QuantType.INT
    scale_factor = torch.tensor(1.0)
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    obj = BiasQuantProxy(quant_type=quant_type,
                         narrow_range=NARROW_RANGE,
                         bit_width=bit_width)
    _, _, input_bit_width = obj(inp, scale_factor, input_bit_width=None)

    assert isinstance(obj.tensor_quant, PrescaledRestrictIntQuant)
    assert input_bit_width == bit_width

@given(bit_width=bit_width_st)
def test_bias_quant_proxy_input_bitwidth(bit_width):
    quant_type = QuantType.INT
    inp = torch.nn.Parameter(torch.randn(SHAPE))
    scale_factor = torch.tensor(1.0)
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    obj = BiasQuantProxy(quant_type=quant_type,
                         narrow_range=NARROW_RANGE,
                         bit_width=None)
    _, _, input_bit_width = obj(inp, scale_factor, bit_width)

    assert isinstance(obj.tensor_quant, PrescaledRestrictIntQuantWithInputBitWidth)
    assert input_bit_width == bit_width

