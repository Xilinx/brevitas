# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import onnxscript
from onnxscript import BOOL
from onnxscript import FLOAT
from onnxscript import INT32
from onnxscript import INT64
from onnxscript import STRING
import torch

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper.clamp import TensorClamp
from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.quant import IntQuant
from brevitas.function import binary_sign
import brevitas.library
from brevitas.quant.solver.common import solve_float_to_int_impl_from_enum

__all__ = [
    "bipolar_quant",
    "bipolar_quant_wrapper",
    "float_quant",
    "float_quant_wrapper",
    "int_quant",
    "int_quant_wrapper",
    "trunc_quant",
    "trunc_quant_wrapper",
    "LIBRARY_STRING",
    "DOMAIN_STRING",
    "DOMAIN_VERSION",]

LIBRARY_STRING = "qonnx"  # Note: if this value is modified, it must also be changed in the QONNXManager classes
DOMAIN_STRING = "qonnx.custom_op.general"
DOMAIN_VERSION = 2
qonnx_op = onnxscript.values.Opset(domain=DOMAIN_STRING, version=DOMAIN_VERSION)


@brevitas.library.custom_op(f"{LIBRARY_STRING}::bipolar_quant", mutates_args=())
def bipolar_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = binary_sign(x) * scale
    return x


@bipolar_quant.register_fake
def _bipolar_quant_fake(tensor_x, scale):
    return torch.empty_like(tensor_x)


@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def BipolarQuant(x: FLOAT, scale: FLOAT) -> FLOAT:
    return x


# We replace bipolar_quant with this function, which wraps to QONNX node we want to generate
@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def bipolar_quant_wrapper(x: FLOAT, scale: FLOAT) -> FLOAT:
    return BipolarQuant(x, scale)


@brevitas.library.custom_op(f"{LIBRARY_STRING}::int_quant", mutates_args=())
def int_quant(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bit_width: torch.Tensor,
        narrow: int,
        signed: int,
        rounding_mode: str) -> torch.Tensor:
    float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
    quant = IntQuant(
        float_to_int_impl=float_to_int_impl(),
        tensor_clamp_impl=TensorClamp(),
        input_view_impl=Identity(),  #TODO: Update this when QONNX support Groupwise export
        narrow_range=torch.tensor(narrow, dtype=x.dtype, device=x.device),
        signed=torch.tensor(signed, dtype=x.dtype, device=x.device))
    x = quant(scale, zero_point, bit_width, x)
    return x


@int_quant.register_fake
def _int_quant_fake(tensor_x, scale, zero_point, bit_width, narrow, signed, rounding_mode):
    return torch.empty_like(tensor_x)


@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def Quant(
        x: FLOAT,
        scale: FLOAT,
        zero_point: FLOAT,
        bit_width: FLOAT,
        narrow: int,
        signed: int,
        rounding_mode: str) -> FLOAT:
    return x


# We replace int_quant with this function, which wraps to QONNX node we want to generate
@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def int_quant_wrapper(
        x: FLOAT,
        scale: FLOAT,
        zero_point: FLOAT,
        bit_width: FLOAT,
        narrow: int,
        signed: int,
        rounding_mode: str) -> FLOAT:
    return Quant(x, scale, zero_point, bit_width, narrow, signed, rounding_mode)


@brevitas.library.custom_op(f"{LIBRARY_STRING}::float_quant", mutates_args=())
def float_quant(
        x: torch.Tensor,
        scale: torch.Tensor,
        exponent_bit_width: torch.Tensor,
        mantissa_bit_width: torch.Tensor,
        exponent_bias: torch.Tensor,
        max_val: torch.Tensor,
        has_inf: int,
        has_nan: int,
        has_subnormal: int,
        rounding_mode: str,
        saturation: int) -> torch.Tensor:
    return x.clone()


@float_quant.register_fake
def _float_quant_fake(
        tensor_x,
        scale,
        exponent_bit_width,
        mantissa_bit_width,
        exponent_bias,
        max_val,
        has_inf,
        has_nan,
        has_subnormal,
        rounding_mode,
        saturation):
    return torch.empty_like(tensor_x)


@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def FloatQuant(
        x: FLOAT,
        scale: FLOAT,
        exponent_bit_width: FLOAT,
        mantissa_bit_width: FLOAT,
        exponent_bias: FLOAT,
        max_val: FLOAT,
        has_inf: int,
        has_nan: int,
        has_subnormal: int,
        rounding_mode: str,
        saturation: int) -> FLOAT:
    return x


# We replace float_quant with this function, which wraps to QONNX node we want to generate
@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def float_quant_wrapper(
        x: FLOAT,
        scale: FLOAT,
        exponent_bit_width: FLOAT,
        mantissa_bit_width: FLOAT,
        exponent_bias: FLOAT,
        max_val: FLOAT,
        has_inf: int,
        has_nan: int,
        has_subnormal: int,
        rounding_mode: str,
        saturation: int) -> FLOAT:
    return FloatQuant(
        x,
        scale,
        exponent_bit_width,
        mantissa_bit_width,
        exponent_bias,
        max_val,
        has_inf,
        has_nan,
        has_subnormal,
        rounding_mode,
        saturation)


@brevitas.library.custom_op(f"{LIBRARY_STRING}::trunc_quant", mutates_args=())
def trunc_quant(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        input_bit_width: torch.Tensor,
        output_scale: torch.Tensor,
        output_bit_width: torch.Tensor,
        rounding_mode: str,
        signed: int,
        narrow: int) -> torch.Tensor:
    # TODO: Restore this (fails when `signed` arg added)
    #float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
    #trunc = TruncIntQuant(
    #    float_to_int_impl=float_to_int_impl(),
    #    bit_width_impl=BitWidthConst(int(output_bit_width)))
    #y_tuple = trunc(x, scale, zero_point, input_bit_width, signed)
    return x.clone()


@trunc_quant.register_fake
def _trunc_quant_fake(
    tensor_x,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    input_bit_width: torch.Tensor,
    output_scale: torch.Tensor,
    output_bit_width: torch.Tensor,
    rounding_mode: str,
    signed: int,
    narrow: int,
):
    return torch.empty_like(tensor_x)


@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def TruncQuant(
        x: FLOAT,
        scale: FLOAT,
        zero_point: FLOAT,
        input_bit_width: FLOAT,
        output_scale: FLOAT,
        output_bit_width: FLOAT,
        rounding_mode: str,
        signed: int,
        narrow: int) -> FLOAT:
    return x


# We replace trunc_quant with this function, which wraps to QONNX node we want to generate
@onnxscript.script(qonnx_op, default_opset=qonnx_op)
def trunc_quant_wrapper(
        x: FLOAT,
        scale: FLOAT,
        zero_point: FLOAT,
        input_bit_width: FLOAT,
        output_scale: FLOAT,
        output_bit_width: FLOAT,
        rounding_mode: str,
        signed: int,
        narrow: int) -> FLOAT:
    return TruncQuant(
        x,
        scale,
        zero_point,
        input_bit_width,
        output_scale,
        output_bit_width,
        rounding_mode,
        signed,
        narrow)
