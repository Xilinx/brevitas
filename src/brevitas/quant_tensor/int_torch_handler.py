# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math
import operator
from typing import Callable
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.utils.torch_utils import compute_channel_view_shape
from brevitas.utils.torch_utils import is_broadcastable
from brevitas.utils.torch_utils import is_fake_mode_active

INT_QUANT_TENSOR_FN_HANDLER = {}


def implements_int_qt(torch_function):

    @functools.wraps(torch_function)
    def decorator(func):
        INT_QUANT_TENSOR_FN_HANDLER[torch_function] = func
        return func

    return decorator


@implements_int_qt(torch.cat)
def cat_handler(*args, **kwargs):
    from brevitas.quant_tensor import IntQuantTensor
    return IntQuantTensor.cat(*args, **kwargs)


@implements_int_qt(F.conv1d)
def conv1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.conv2d)
def conv2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.conv3d)
def conv3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.conv_transpose1d)
def conv_transpose1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.conv_transpose2d)
def conv_transpose2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.conv_transpose3d)
def conv_transpose3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.linear)
def linear_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.linear, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_int_qt(F.embedding)
def embedding_handler(input, quant_weight, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import IntQuantTensor

    quant_weight_value = _unpack_quant_tensor(quant_weight)
    out = F.embedding(input, quant_weight_value, *args, **kwargs)

    if isinstance(quant_weight, IntQuantTensor):
        scale = quant_weight.scale
        zero_point = quant_weight.zero_point
        bit_width = quant_weight.bit_width
        if any(t.numel() > 1 for t in [scale, zero_point, bit_width]):
            raise RuntimeError("Only per-tensor quantization is supported.")
        signed = quant_weight.signed
        training = quant_weight.training
        out = IntQuantTensor(out, scale, zero_point, bit_width, signed, training)
    return out


@implements_int_qt(F.avg_pool2d)
def avg_pool2d_handler(
        quant_input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    value, scale, bit_width = avg_pool2d_impl(
        quant_input.value,
        quant_input.scale,
        quant_input.bit_width,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override)
    return quant_input.set(value=value, scale=scale, bit_width=bit_width)


@implements_int_qt(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d_handler(quant_input, output_size):
    value, scale, bit_width = adaptive_avg_pool2d_impl(
        quant_input.value, quant_input.scale, quant_input.bit_width, output_size)
    return quant_input.set(value=value, scale=scale, bit_width=bit_width)


def avg_pool2d_impl(
        value,
        scale,
        bit_width,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override):
    value = F.avg_pool2d(
        value, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    if divisor_override is not None:
        avg_scaling = divisor_override
    elif isinstance(kernel_size, tuple):
        avg_scaling = kernel_size[0] * kernel_size[1]
    else:
        avg_scaling = kernel_size * kernel_size
    return value, scale / avg_scaling, max_acc_bit_width_avg_pool2d(bit_width, avg_scaling)


def adaptive_avg_pool2d_impl(value, scale, bit_width, output_size):
    pooled = F.adaptive_avg_pool2d(value, output_size)
    kernel_size, _ = compute_kernel_size_stride(value.shape[-2:], pooled.shape[-2:])
    avg_scaling = functools.reduce(operator.mul, kernel_size, 1)
    return pooled, scale / avg_scaling, max_acc_bit_width_avg_pool2d(bit_width, avg_scaling)


def compute_kernel_size_stride(input_shape, output_shape):
    kernel_size_list = []
    stride_list = []
    for inp, out in zip(input_shape, output_shape):
        stride = inp // out
        kernel_size = inp - (out - 1) * stride
        kernel_size_list.append(kernel_size)
        stride_list.append(stride)
    return kernel_size_list, stride_list


def quant_layer(fn, quant_input, quant_weight, bias, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import IntQuantTensor

    output_scale = None
    output_bit_width = None
    output_zero_point = None
    output_signed = None
    max_acc_bit_width = FN_ACC_BITWIDTH_MAPPING[fn]

    compute_output_quant_tensor = isinstance(quant_input, IntQuantTensor) and isinstance(
        quant_weight, IntQuantTensor)

    if bias is None:
        output = fn(
            _unpack_quant_tensor(quant_input),
            _unpack_quant_tensor(quant_weight),
            None,
            *args,
            **kwargs)
    else:
        output = fn(
            _unpack_quant_tensor(quant_input),
            _unpack_quant_tensor(quant_weight),
            _unpack_quant_tensor(bias),
            *args,
            **kwargs)

    if isinstance(quant_input, IntQuantTensor) and isinstance(quant_weight, IntQuantTensor):
        output_bit_width = max_acc_bit_width(
            quant_input.bit_width,
            quant_weight.bit_width,
            quant_weight.value.shape,
            *args,
            **kwargs)
        output_scale = quant_output_scale_impl(
            fn, quant_input.value, quant_input.scale, quant_weight.scale)
        output_signed = quant_input.signed or quant_weight.signed
        output_training = quant_input.training or quant_weight.training

    if bias is not None:
        if output_scale is not None:
            channel_dim = -1 if isinstance(fn, torch.nn.Linear) else 1
            output_scale_broadcast_shape = compute_channel_view_shape(
                quant_input, channel_dim=channel_dim)
            output_zero_point = -_unpack_quant_tensor(bias).view(
                output_scale_broadcast_shape) / output_scale
            if isinstance(bias, IntQuantTensor):
                # A bias quantized with the scale of the accumulator needs no zero point. This is
                # selected with torch.where, since the scales cannot be compared while tracing
                bias_on_accumulator_grid = torch.isclose(bias.scale, output_scale).all()
                output_zero_point = torch.where(
                    bias_on_accumulator_grid,
                    torch.zeros_like(output_zero_point),
                    output_zero_point)
        if output_bit_width is not None and isinstance(bias, IntQuantTensor):
            output_bit_width = torch.where(
                bias.bit_width > output_bit_width, bias.bit_width, output_bit_width)
            output_bit_width = output_bit_width + 1

    if compute_output_quant_tensor:
        if is_fake_mode_active():
            # The zero points cannot be read while tracing, so the worst case is assumed
            warnings.warn(
                "Cannot check that the zero points are zero while tracing with torch.compile or "
                "torch.export, where tensor values are unknown. Returning Tensor.")
            compute_output_quant_tensor = False
        elif (quant_input.zero_point != 0.0).any() or (quant_weight.zero_point != 0.0).any():
            warnings.warn("Computing zero point of output accumulator not supported yet.")
            compute_output_quant_tensor = False
        if output_scale is None:
            warnings.warn("Could not compute output scale factor, returning Tensor")
            compute_output_quant_tensor = False

    if compute_output_quant_tensor:
        if output_zero_point is None:
            output_zero_point = torch.zeros(1).type_as(output)
        return create_int_quant_tensor(
            output,
            output_scale,
            output_bit_width,
            output_zero_point,
            output_signed,
            output_training)
    else:
        return output


def create_int_quant_tensor(tensor, scale, bit_width, zero_point, signed, training):
    from brevitas.quant_tensor import IntQuantTensor
    return IntQuantTensor(
        tensor,
        scale=scale,
        zero_point=zero_point,
        bit_width=bit_width,
        signed=signed,
        training=training)


def quant_output_scale_impl(
        fn: Callable, inp: Tensor, quant_input_scale: Tensor, quant_weight_scale: Tensor):
    channel_dim = -1 if fn == F.linear else 1
    output_scale_shape = compute_channel_view_shape(inp, channel_dim=channel_dim)

    quant_weight_scale = quant_weight_scale.view(output_scale_shape)
    quant_input_scale = quant_input_scale.view(output_scale_shape)
    if not is_broadcastable(quant_weight_scale.shape, quant_input_scale.shape):
        return None

    output_scale = quant_weight_scale * quant_input_scale
    return output_scale


def max_acc_bit_width_convNd(input_bit_width, weight_bit_width, weight_shape, *args, **kwargs):
    max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
    max_kernel_val = max_int(bit_width=weight_bit_width, signed=False, narrow_range=False)
    in_channel = weight_shape[1]
    kernel_size = math.prod(weight_shape[2:])
    max_uint_output = max_uint_input * max_kernel_val * kernel_size * in_channel
    max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
    return max_output_bit_width


def max_acc_bit_width_linear(input_bit_width, weight_bit_width, weight_shape, *args, **kwargs):
    max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
    max_kernel_val = max_int(bit_width=weight_bit_width, signed=False, narrow_range=False)
    in_channel = weight_shape[1]
    max_uint_output = max_uint_input * max_kernel_val * in_channel
    max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
    return max_output_bit_width


def max_acc_bit_width_convtransposeNd(
        input_bit_width, weight_bit_width, weight_shape, *args, **kwargs):
    stride = kwargs['stride']
    max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
    max_kernel_val = max_int(bit_width=weight_bit_width, signed=False, narrow_range=False)
    out_channel = weight_shape[1]
    kernel_shape = weight_shape[2:]

    patch_size = 0
    for s, k in zip(stride, kernel_shape):
        patch_size += max(math.ceil(k / s), 1)

    max_uint_output = max_uint_input * max_kernel_val * patch_size * out_channel
    max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
    return max_output_bit_width


def max_acc_bit_width_avg_pool2d(input_bit_width, avg_scaling):
    max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
    max_uint_output = max_uint_input * avg_scaling
    max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
    return max_output_bit_width


FN_ACC_BITWIDTH_MAPPING = {
    F.linear: max_acc_bit_width_linear,
    F.conv1d: max_acc_bit_width_convNd,
    F.conv2d: max_acc_bit_width_convNd,
    F.conv3d: max_acc_bit_width_convNd,
    F.conv_transpose1d: max_acc_bit_width_convtransposeNd,
    F.conv_transpose2d: max_acc_bit_width_convtransposeNd,
    F.conv_transpose3d: max_acc_bit_width_convtransposeNd,
    F.avg_pool2d: max_acc_bit_width_avg_pool2d}
