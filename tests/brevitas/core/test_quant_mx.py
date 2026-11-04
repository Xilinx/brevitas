# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
MX quantizers (MXFP and MXINT) against golden references of the OCP MX specification
"""

import struct
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from hypothesis import assume
from hypothesis import example
from hypothesis import given
import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas.core.scaling import RoundMidMaxSte
from brevitas.export.inference import quant_inference_mode
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_linear import QuantLinear
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.mx_quant_ocp import MXInt8WeightMSE
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8Weight
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8WeightMSE
from tests.brevitas.hyp_helper import float_tensor_nz_st

torch.manual_seed(0)


# debug utility
def to_string(val: Union[torch.Tensor, float],
              spaced: bool = True,
              code: str = "f") -> Union[str, List[str]]:
    """ Debug util for visualizing float values """

    def scalar_to_string(val: float, spaced: bool) -> str:
        s = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!' + code, val))
        spaced = spaced and len(s) == 32
        return f"{s[0]} {s[1:9]} {s[9:]}" if spaced else s

    if isinstance(val, float):
        return scalar_to_string(val, spaced)
    val = val.view(-1)
    return [scalar_to_string(val[i].item(), spaced) for i in range(val.numel())]


# debug utility
def check_bits(val: Union[torch.Tensor, float], mbits: int) -> Tuple[bool, int]:
    """ return (too many precision bits, lowest mantissa bit) """
    strings = to_string(val, spaced=False)
    if isinstance(strings, str):
        strings = [strings]
    error, lowest = False, 0
    for s in strings:
        mant = s[9:]
        error = error or "1" in mant[mbits:]
        lowest = max(lowest, mant.find("1"))
    return error, lowest


# Avoid returning exp 0 if we is 0
def safe_frexp(x: torch.Tensor) -> torch.Tensor:
    """torch.frexp returns unbiased exponent 0 for 0.0, which is not what we want."""
    if x.is_cuda and x.dtype not in (torch.float32, torch.float16):
        x = x.float()  # no gpu support for frexp on bfloat16 or any float8
    return torch.where(x == 0.0, torch.tensor(-126, dtype=torch.int32), x.frexp().exponent - 1)


class MXFP:
    """
    MXFP - Quantize OCP MXFP floating point types.
    A type is defined as ebits, mbits, bias, and inf/nan handling.
    """
    CONFIG = dict(
        e5m2=(5, 2, 15, "ieee"),
        e4m3=(4, 3, 7, "fn"),
        e3m2=(3, 2, 3, "fnuz"),
        e2m3=(2, 3, 1, "fnuz"),
        e2m1=(2, 1, 1, "fnuz"))

    def __init__(self, name, tile_size: Optional[int] = 32):
        self.name = name.lower()
        assert self.name in self.CONFIG
        self.ebits, self.mbits, self.bias, self.infnan = self.CONFIG[self.name]
        self.tile_size = tile_size

    @property  # maximum unbiased exponent for this type
    def emax(self) -> int:
        return 2 ** self.ebits - 1 - self.bias - int(self.infnan == "ieee")

    @property  # minimum unbiased exponent for this type
    def emin(self) -> int:
        return 1 - self.bias

    @property  # maximum representable value; the "fn" reserves values for all non-sign bits == 1
    def maxval(self) -> float:
        return 2 ** self.emax * (2.0 - (1 + int(self.infnan == "fn")) * 2 ** (-self.mbits))

    @property  # for alternative scale selection
    def midmax(self) -> float:
        return (2 ** (self.emax + 1) - self.maxval) / 2. + self.maxval

    @property  # minimum representable positive value
    def minval(self) -> float:
        return 2 ** self.emin * 2 ** (-self.mbits)

    def quantize(self, tensor: torch.Tensor, axis: int = -1, select: bool = False):
        """
        Fake quantize along the indicated dimension. This method assumes the tile dimension is the size of the tile,
        so some reshaping and possibly padding is likely required.  From there, we have 5 needed lines of code.
        """
        exp = safe_frexp(tensor)  # safe_frexp pretends the mantissa is < 1.0
        shared = exp.amax(axis, keepdim=True)  # shared exponent per the OCP MX spec

        # This is an alternative to the OCP MX scale selection, which chooses the maximum exponent (maxexp).
        # Instead, choose maxexp + 1 if absmax is closer to 2^(maxexp+1) than maxval. This reduces error on
        # the highest magnitude value at the potential cost increased error or underflow of the smallest.
        # Ad hoc MSE test shows that e4m3, due to reserving the most significant value for Nan, benefits the
        # most from this technique.  In hardware or a kernel, this is as simple as comparing bits [30:21]
        # instead of [30:23] when getting max exponent, then add 1 to the max eeeeeeeemm and shift right two.
        #             e2m1    e3m2    e2m3    e4m3    e5m2
        #   max     0.01325 0.00291 0.00080 0.00085 0.00291
        #   best    0.01254 0.00280 0.00079 0.00071 0.00280

        if select:
            midmax = self.midmax * (shared - self.emax).exp2()
            shared[tensor.abs().amax(axis, keepdim=True) > midmax] += 1

        # The way this works is to appropriately shift values so that rounding can work, then shift them back.
        # All values that are representable as normal given the scale are shifted up by the difference
        # between the individual exponent and zero, plus the mantissa width.  Subnormals get the same,
        # but with decreasing mantissa bits.  The maxval for saturation is adjusted on a per block basis.
        scale = (self.mbits - (shared - exp - (self.emax - self.emin)).clamp_min(0) - exp).exp2()
        # about that last line of code:
        # The "offset" is the number of mbits lost to subnormal/underflow. This is based on the difference between
        # the shared exponent and the individual exponent, adjusted to the dynamic range of normals for this type.
        # It can't be negative, because we subtract it from mbits, and don't want to exceed the available mbits.
        #   offset = (shared - exp - (self.emax - self.emin)).clamp_min(0)
        # The shift left will be mbits - offset - exp, which for negative exponents gets them into the right range.
        maxval = self.maxval * (shared - self.emax).exp2()  # scale maxval per tile
        return ((tensor * scale).round() / scale).clamp(-maxval, maxval)


MAP = {"e4m3": (4, 3), "e5m2": (5, 2), "e2m3": (2, 3), "e3m2": (3, 2), "e2m1": (2, 1)}
SCALE_ROUNDING = ["floor", "midmax"]
RANGE = 1e9

edge_cases = [
    torch.tensor([[7.5 if i == 0 else 1.0 for i in range(32)]]),  # MXFP4, MidMax
    torch.tensor([[3.875 if i == 0 else 1.0 for i in range(32)]]),  # MXFP8E4M3, MidMax
]


@example(inp=edge_cases[0])
@example(inp=edge_cases[1])
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('bit_widths', list(MAP.keys()))
@pytest_cases.parametrize('scale_rounding', SCALE_ROUNDING)
def test_act_mx(inp, bit_widths, scale_rounding):
    torch.set_printoptions(precision=12, sci_mode=False)
    exp, mant = MAP[bit_widths]

    extra_kwargs = {}
    # Default rounding should be 'floor', so only create this dict if we are overriding the default
    if scale_rounding == "midmax":
        extra_kwargs["restrict_value_float_to_int_impl"] = RoundMidMaxSte
    act_quant = QuantIdentity(
        MXFloat8e4m3Act,
        exponent_bit_width=exp,
        mantissa_bit_width=mant,
        bit_width=mant + exp + 1,
        group_dim=1,
        return_quant_tensor=True,
        **extra_kwargs)
    act_quant.eval()
    x = inp

    quantizer = MXFP(bit_widths)

    qx = act_quant(x)

    y = quantizer.quantize(x, select=scale_rounding == "midmax")
    assert torch.allclose(qx.value, y, atol=1e-8)


@example(inp=edge_cases[0])
@example(inp=edge_cases[1])
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('bit_widths', list(MAP.keys()))
@pytest_cases.parametrize('scale_rounding', SCALE_ROUNDING)
@pytest_cases.parametrize('weight_quant_type', ['stats', 'parameter_from_stats'])
def test_weight_mx(inp, bit_widths, scale_rounding, weight_quant_type):
    torch.set_printoptions(precision=12, sci_mode=False)
    exp, mant = MAP[bit_widths]
    extra_kwargs = {}
    # Default rounding should be 'floor', so only create this dict if we are overriding the default
    if scale_rounding == "midmax":
        extra_kwargs["weight_restrict_value_float_to_int_impl"] = RoundMidMaxSte
    weight_quant = QuantLinear(
        32,
        1,
        bias=False,
        weight_quant=MXFloat8e4m3Weight,
        weight_scaling_impl_type=weight_quant_type,
        weight_exponent_bit_width=exp,
        weight_mantissa_bit_width=mant,
        weight_bit_width=mant + exp + 1,
        **extra_kwargs)

    x = inp
    weight_quant.weight.data = x
    weight_quant.weight_quant.init_tensor_quant()
    quantizer = MXFP(bit_widths)

    qx_weight = weight_quant.quant_weight()
    qx_weight_two = weight_quant.quant_weight()

    y = quantizer.quantize(x, select=scale_rounding == "midmax")
    assert torch.allclose(qx_weight.value, y, atol=1e-8)
    assert torch.allclose(qx_weight_two.value, y, atol=1e-8)


@pytest_cases.parametrize(
    'shape, group_dim_pos, group_dim_neg', [((1, 32), 1, -1), ((2, 32), 1, -1), ((2, 3, 32), 2, -1),
                                            ((2, 3, 32), 1, -2)])
def test_act_mx_negative_group_dim(shape, group_dim_pos, group_dim_neg):
    """Verify that negative group_dim values produce the same results as positive equivalents."""
    torch.manual_seed(0)
    x = torch.randn(shape)

    def make_act_quant(group_dim):
        return QuantIdentity(
            MXFloat8e4m3Act,
            exponent_bit_width=4,
            mantissa_bit_width=3,
            bit_width=8,
            group_dim=group_dim,
            return_quant_tensor=True)

    act_pos = make_act_quant(group_dim_pos)
    act_pos.eval()
    act_neg = make_act_quant(group_dim_neg)
    act_neg.eval()

    qx_pos = act_pos(x)
    qx_neg = act_neg(x)

    assert torch.allclose(qx_pos.value, qx_neg.value, atol=1e-8)


class MXINT:
    """
    MXINT - Quantize to OCP MX integer types (OCP MX v1.0, section 6.3, INT8 elements as in Table 6).
    The element has 1 sign bit, 1 integer bit and (bit_width - 2) fractional bits, so the largest
    power of two it represents is 2^0 and the shared scale is X = 2^floor(log2(max|V|)).
    """

    def __init__(self, bit_width: int = 8, narrow_range: bool = False):
        self.frac_bits = bit_width - 2  # INT8: implicit scale 2^-6
        self.max = 2 ** (bit_width - 1) - 1  # INT8: +1 63/64 in element units
        # The maximum negative representation (-2 for INT8) may be left unused
        self.min = -self.max if narrow_range else -self.max - 1

    def step(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """ Size of one integer step of the block, X * 2^-frac_bits """
        shared = safe_frexp(x).amax(axis, keepdim=True)  # floor(log2(max|V|)), exact
        return (shared - self.frac_bits).float().exp2()

    def quantize(self, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        step = self.step(x, axis)
        # torch.round is round half to even; out of range values saturate, preserving the sign
        return torch.clamp(torch.round(x / step), self.min, self.max) * step


SCALING_MIN_VAL = 1e-10  # Default scaling_min_val of the MX quantizers


def assume_ocp_comparable(x: torch.Tensor, axis: int = -1):
    """
    Exclude blocks that Brevitas does not quantize as in the OCP spec for reasons that are out of scope
    of the golden reference: block maxima below scaling_min_val, and block maxima for which log2 in FP32
    rounds up to the next power of two (see test_mx_shared_exponent_exact).
    """
    amax = x.abs().amax(axis)
    assume(bool((amax > SCALING_MIN_VAL).all()))
    assume(bool((torch.log2(amax).floor() == safe_frexp(amax)).all()))


def mxint_act_quant(quant=MXInt8Act, **kwargs) -> QuantIdentity:
    act_quant = QuantIdentity(quant, group_dim=1, return_quant_tensor=True, **kwargs)
    act_quant.eval()
    return act_quant


def mxint_weight_layer(weight: torch.Tensor, quant=MXInt8Weight, **kwargs) -> QuantLinear:
    layer = QuantLinear(weight.shape[1], weight.shape[0], bias=False, weight_quant=quant, **kwargs)
    layer.weight.data = weight
    layer.weight_quant.init_tensor_quant()
    return layer


# Quantizers are injectors, which pytest_cases can not take as parameter values: index them by name
MXINT_WEIGHT_QUANTIZERS = {
    'mxint8': MXInt8Weight,
    'mxint8_mse': MXInt8WeightMSE,
    'shifted_mxuint8': ShiftedMXUInt8Weight,
    'shifted_mxuint8_mse': ShiftedMXUInt8WeightMSE}


def skip_mse_under_jit(quant_name):
    if config.JIT_ENABLED and quant_name.endswith('_mse'):
        pytest.skip("MSE scaling requires JIT to be disabled")


MXINT_WEIGHT_SCALING = ['stats', 'parameter_from_stats']
TOP_OF_BINADE_BLOCK = torch.tensor([[1.9] + [0.1] * 31])

mxint_edge_cases = [
    TOP_OF_BINADE_BLOCK,  # amax at the top of its binade
    torch.tensor([[1.9 * 2 ** 10] + [-3.0] * 31]),  # amax at the top of its binade, scaled up
    torch.tensor([[1.995] + [-1.995] + [0.5] * 30]),  # +-127.68 steps: saturates to 127 / -128
    torch.tensor([[4.0] + [0.3] * 31]),  # amax an exact power of two
    torch.tensor([[-8.0] + [0.3] * 31]),  # negative amax an exact power of two
    torch.tensor([[1.0] + [2.5 / 64, 3.5 / 64, -2.5 / 64, -0.5 / 64] * 7 + [0.5] * 3]),  # ties
    torch.tensor([[0.75, -1.3, 0.02, -0.6] * 8]),  # mixed signs
]


def mxint_examples(fn):
    for case in mxint_edge_cases:
        fn = example(inp=case)(fn)
    return fn


@pytest_cases.parametrize('weight_scaling', MXINT_WEIGHT_SCALING)
def test_mxint_top_binade_not_clipped(weight_scaling):
    """ A block maximum in the top half of its binade must not be clipped (OCP MXINT8: 1.90625) """
    x = TOP_OF_BINADE_BLOCK
    expected = MXINT().quantize(x)
    assert expected[0, 0].item() == 1.90625

    qx_act = mxint_act_quant()(x)
    assert qx_act.value[0, 0].item() == 1.90625

    layer = mxint_weight_layer(x, weight_scaling_impl_type=weight_scaling)
    qx_weight = layer.quant_weight()
    assert qx_weight.value[0, 0].item() == 1.90625


@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('weight_scaling', [None] + MXINT_WEIGHT_SCALING)
def test_mxint_error_bounded_by_one_step(inp, weight_scaling):
    """ Rounding plus saturation at 127 never moves an element by more than one integer step """
    assume_ocp_comparable(inp)
    if weight_scaling is None:
        qx = mxint_act_quant()(inp).value
    else:
        qx = mxint_weight_layer(inp, weight_scaling_impl_type=weight_scaling).quant_weight().value
    step = MXINT().step(inp)
    assert ((qx - inp).abs() <= step).all()


@mxint_examples
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('narrow_range', [False, True])
def test_act_mxint(inp, narrow_range):
    assume_ocp_comparable(inp)
    qx = mxint_act_quant(narrow_range=narrow_range)(inp)
    y = MXINT(narrow_range=narrow_range).quantize(inp)
    assert torch.equal(qx.value, y)


@mxint_examples
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('narrow_range', [False, True])
@pytest_cases.parametrize('weight_scaling', MXINT_WEIGHT_SCALING)
def test_weight_mxint(inp, narrow_range, weight_scaling):
    assume_ocp_comparable(inp)
    layer = mxint_weight_layer(
        inp, weight_scaling_impl_type=weight_scaling, weight_narrow_range=narrow_range)
    y = MXINT(narrow_range=narrow_range).quantize(inp)
    assert torch.equal(layer.quant_weight().value, y)
    assert torch.equal(layer.quant_weight().value, y)


@mxint_examples
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
def test_mxint_inference_mode(inp):
    """ The inference handlers (used for evaluation and export) match the OCP reference too """
    assume_ocp_comparable(inp)
    y = MXINT().quantize(inp)

    act_quant = QuantIdentity(MXInt8Act, group_dim=1)
    act_quant(inp)
    act_quant.eval()
    with quant_inference_mode(act_quant, compile=False):
        act_quant(inp)
        assert torch.equal(act_quant(inp), y)

    layer = mxint_weight_layer(inp)
    layer.eval()
    with quant_inference_mode(layer, compile=False):
        layer(inp)
        qx_weight = layer.quant_weight()
        qx_weight = qx_weight.value if hasattr(qx_weight, 'value') else qx_weight
        assert torch.equal(qx_weight, y)


@pytest_cases.parametrize('quant_name', ['mxint8_mse', 'shifted_mxuint8', 'shifted_mxuint8_mse'])
def test_mxint_mse_and_shifted_top_binade_not_clipped(quant_name):
    """ No golden reference for these (not OCP formats), but the block maximum must not be clipped """
    skip_mse_under_jit(quant_name)
    x = TOP_OF_BINADE_BLOCK
    qx = mxint_weight_layer(x, quant=MXINT_WEIGHT_QUANTIZERS[quant_name]).quant_weight()
    assert (qx.value[0, 0] - x[0, 0]).abs() <= qx.scale.max()


@mxint_examples
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
def test_shifted_mxuint_error_bounded(inp):
    """
    The block range spans at most 255 integer steps, so only the rounding of the zero-point can push
    the block maximum out of range: the error stays within 1.5 steps.
    """
    min_val = inp.amin(-1, keepdim=True).clamp(max=0.)
    block_range = inp.amax(-1, keepdim=True) - min_val
    assume(bool((block_range > SCALING_MIN_VAL).all()))
    assume(bool((torch.log2(block_range).floor() == safe_frexp(block_range)).all()))
    qx = mxint_weight_layer(inp, quant=ShiftedMXUInt8Weight).quant_weight()
    scale = qx.scale.view(-1, 1)
    assert (block_range / scale < 256).all()
    assert ((qx.value - inp).abs() <= 1.5 * scale).all()


@mxint_examples
@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=RANGE, min_val=-RANGE))
@pytest_cases.parametrize('base_quant_name', ['mxint8', 'shifted_mxuint8'])
def test_mxint_mse_not_worse_than_max(inp, base_quant_name):
    """ The MSE scale search may clip, but only when it lowers the error w.r.t. max-based scaling """
    skip_mse_under_jit(base_quant_name + '_mse')
    assume_ocp_comparable(inp)
    base_quant = MXINT_WEIGHT_QUANTIZERS[base_quant_name]
    mse_quant = MXINT_WEIGHT_QUANTIZERS[base_quant_name + '_mse']
    mse_error = (mxint_weight_layer(inp, quant=mse_quant).quant_weight().value - inp).pow(2).sum()
    base_error = (mxint_weight_layer(inp, quant=base_quant).quant_weight().value - inp).pow(2).sum()
    assert mse_error <= base_error * (1 + 1e-6)


SHARED_EXPONENT_QUANTIZERS = {'mxfloat8': (MXFloat8e4m3Act, lambda x: MXFP('e4m3').quantize(x))}


@pytest.mark.xfail(
    strict=True,
    reason="Shared exponent computed as floor(log2(amax)) in the input dtype, which rounds up for "
    "values just below a power of two (OCP requires the largest power of two <= max|V|)")
@pytest_cases.parametrize('quant_name', SHARED_EXPONENT_QUANTIZERS.keys())
def test_mx_shared_exponent_exact(quant_name):
    quant, reference = SHARED_EXPONENT_QUANTIZERS[quant_name]
    amax = torch.nextafter(torch.tensor(2. ** 8), torch.tensor(0.))
    assert safe_frexp(amax).item() == 7
    x = torch.tensor([[amax.item(), 86.0] + [0.1] * 30])
    qx = mxint_act_quant(quant=quant)(x)
    assert torch.allclose(qx.value, reference(x), atol=1e-8)
