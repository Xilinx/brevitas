# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math
import warnings

import torch
import torch.nn.functional as F

import brevitas
from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.utils.torch_utils import compute_channel_view_shape

QUANT_TENSOR_FN_HANDLER = {}


def implements(torch_function):

    @functools.wraps(torch_function)
    def decorator(func):
        QUANT_TENSOR_FN_HANDLER[torch_function] = func
        return func

    return decorator


def quant_invariant_handler(fn, inp, *args, **kwargs):
    out_value = fn(inp.value, *args, **kwargs)
    return inp.set(value=out_value)


@implements(torch.flatten)
def flatten_handler(inp, *args, **kwargs):
    return inp.flatten(*args, **kwargs)


@implements(torch.reshape)
def reshape_handler(inp, *args, **kwargs):
    return inp.reshape(*args, **kwargs)


@implements(torch.transpose)
def transpose_handler(inp, *args, **kwargs):
    return inp.transpose(*args, **kwargs)

@implements(torch.permute)
def permute_handler(inp, *args, **kwargs):
    return inp.permute(*args, **kwargs)

@implements(torch.squeeze)
def squeeze_handler(inp, *args, **kwargs):
    return inp.squeeze(*args, **kwargs)

@implements(torch.unsqueeze)
def unsqueeze_handler(inp, *args, **kwargs):
    return inp.unsqueeze(*args, **kwargs)

@implements(torch.cat)
def cat_handler(*args, **kwargs):
    from brevitas.quant_tensor import QuantTensor
    return QuantTensor.cat(*args, **kwargs)


@implements(F.pad)
def pad_handler(*args, **kwargs):
    # TODO check padding value is legal
    return quant_invariant_handler(F.pad, *args, **kwargs)


@implements(F.relu)
def relu_qt_handler(*args, **kwargs):
    return quant_invariant_handler(F.relu, *args, **kwargs)


@implements(F.relu6)
def relu6_qt_handler(*args, **kwargs):
    return quant_invariant_handler(F.relu6, *args, **kwargs)


@implements(F.hardtanh)
def hardtanh_qt_handler(*args, **kwargs):
    return quant_invariant_handler(F.hardtanh, *args, **kwargs)


@implements(F.alpha_dropout)
def alpha_dropout_handler(*args, **kwargs):
    return quant_invariant_handler(F.alpha_dropout, *args, **kwargs)


@implements(F.dropout)
def dropout_handler(*args, **kwargs):
    return quant_invariant_handler(F.dropout, *args, **kwargs)


@implements(F.dropout2d)
def dropout2d_handler(*args, **kwargs):
    return quant_invariant_handler(F.dropout2d, *args, **kwargs)


@implements(F.dropout3d)
def dropout3d_handler(*args, **kwargs):
    return quant_invariant_handler(F.dropout3d, *args, **kwargs)


@implements(F.max_pool1d)
def max_pool1d(*args, **kwargs):
    return quant_invariant_handler(F.max_pool1d, *args, **kwargs)


@implements(F.max_pool2d)
def max_pool2d_handler(*args, **kwargs):
    return quant_invariant_handler(F.max_pool2d, *args, **kwargs)


@implements(F.max_pool3d)
def max_pool3d_handler(*args, **kwargs):
    return quant_invariant_handler(F.max_pool3d, *args, **kwargs)


@implements(F.adaptive_max_pool1d)
def adaptive_max_pool1d_handler(*args, **kwargs):
    return quant_invariant_handler(F.adaptive_max_pool1d, *args, **kwargs)


@implements(F.adaptive_max_pool2d)
def adaptive_max_pool2d_handler(*args, **kwargs):
    return quant_invariant_handler(F.adaptive_max_pool2d, *args, **kwargs)


@implements(F.adaptive_max_pool3d)
def adaptive_max_pool3d_handler(*args, **kwargs):
    return quant_invariant_handler(F.adaptive_max_pool3d, *args, **kwargs)


@implements(F.interpolate)
def interpolate_handler(
        inp,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=None,
        recompute_scale_factor=None,
        **kwargs):  # support newer kwargs added in recent pytorch versions
    if mode == 'nearest' or mode == 'nearest_exact':
        return quant_invariant_handler(
            F.interpolate,
            inp,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            **kwargs)
    else:
        return F.interpolate(
            inp.value,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            **kwargs)


@implements(F.pixel_shuffle)
def pixel_shuffle_handler(*args, **kwargs):
    return quant_invariant_handler(F.pixel_shuffle, *args, **kwargs)


@implements(F.pixel_unshuffle)
def pixel_unshuffle_handler(*args, **kwargs):
    return quant_invariant_handler(F.pixel_unshuffle, *args, **kwargs)


@implements(F.conv1d)
def conv1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.conv2d)
def conv2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.conv3d)
def conv3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.conv_transpose1d)
def conv_transpose1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.conv_transpose2d)
def conv_transpose2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.conv_transpose3d)
def conv_transpose3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.linear)
def linear_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.linear, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements(F.embedding)
def embedding_handler(input, quant_weight, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import QuantTensor

    quant_weight_value = _unpack_quant_tensor(quant_weight)
    out = F.embedding(input, quant_weight_value, *args, **kwargs)

    if isinstance(quant_weight, QuantTensor):
        scale = quant_weight.scale
        zero_point = quant_weight.zero_point
        bit_width = quant_weight.bit_width
        if any(t.numel() > 1 for t in [scale, zero_point, bit_width]):
            raise RuntimeError("Only per-tensor quantization is supported.")
        signed = quant_weight.signed
        training = quant_weight.training
        out = QuantTensor(out, scale, zero_point, bit_width, signed, training)
    return out


@implements(F.avg_pool2d)
def avg_pool2d_handler(
        quant_input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    from brevitas.quant_tensor import _unpack_quant_tensor

    x = F.avg_pool2d(
        _unpack_quant_tensor(quant_input),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override)

    max_acc_bit_width = FN_ACC_BITWIDTH_MAPPING[F.avg_pool2d]
    # remove avg scaling
    if isinstance(kernel_size, tuple):
        avg_scaling = kernel_size[0] * kernel_size[1]
    else:
        avg_scaling = kernel_size * kernel_size
    rescaled_value = x * avg_scaling
    quant_input = quant_input.set(value=rescaled_value)
    quant_input = quant_input.set(bit_width=max_acc_bit_width(quant_input.bit_width, avg_scaling))
    return quant_input


@implements(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d_handler(quant_input, output_shape):
    from functools import reduce
    from operator import mul

    from brevitas.nn.quant_avg_pool import TruncAdaptiveAvgPool2d
    from brevitas.quant_tensor import _unpack_quant_tensor

    x = F.adaptive_avg_pool2d(_unpack_quant_tensor(quant_input), output_shape)
    k_size, stride = TruncAdaptiveAvgPool2d.compute_kernel_size_stride(quant_input.value.shape[2:], x.shape[2:])

    max_acc_bit_width = FN_ACC_BITWIDTH_MAPPING[F.avg_pool2d]
    reduce_size = reduce(mul, k_size, 1)
    rescaled_value = x * reduce_size  # remove avg scaling

    quant_input = quant_input.set(value=rescaled_value)
    quant_input = quant_input.set(bit_width=max_acc_bit_width(quant_input.bit_width, reduce_size))
    return quant_input


def quant_layer(fn, quant_input, quant_weight, bias, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import QuantTensor

    output_scale = None
    output_bit_width = None
    output_zero_point = None
    output_signed = None
    max_acc_bit_width = FN_ACC_BITWIDTH_MAPPING[fn]

    compute_output_quant_tensor = isinstance(quant_input, QuantTensor) and isinstance(
        quant_weight, QuantTensor)

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

    if isinstance(quant_input, QuantTensor) and isinstance(quant_weight, QuantTensor):
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
            if (isinstance(bias, QuantTensor) and
                    not torch.allclose(bias.scale, output_scale)) or not isinstance(bias,
                                                                                    QuantTensor):
                channel_dim = -1 if isinstance(fn, torch.nn.Linear) else 1
                output_scale_broadcast_shape = compute_channel_view_shape(
                    quant_input, channel_dim=channel_dim)
                output_zero_point = -_unpack_quant_tensor(bias).view(
                    output_scale_broadcast_shape) / output_scale
        if output_bit_width is not None and isinstance(bias, QuantTensor):
            output_bit_width = torch.where(
                bias.bit_width > output_bit_width, bias.bit_width, output_bit_width)
            output_bit_width = output_bit_width + 1

    if compute_output_quant_tensor:
        if (isinstance(quant_input, QuantTensor) and
            (quant_input.zero_point != 0.0).any()) or (isinstance(quant_weight, QuantTensor) and
                                                       (quant_weight.zero_point != 0.0).any()):
            warnings.warn("Computing zero point of output accumulator not supported yet.")
            compute_output_quant_tensor = False

    if compute_output_quant_tensor:
        if output_zero_point is None:
            output_zero_point = torch.zeros(1).type_as(output)

        return create_quant_tensor(
            output,
            output_scale,
            output_bit_width,
            output_zero_point,
            output_signed,
            output_training)
    else:
        return output


def create_quant_tensor(tensor, scale, bit_width, zero_point, signed, training):
    from brevitas.quant_tensor import QuantTensor
    return QuantTensor(
        tensor,
        scale=scale,
        zero_point=zero_point,
        bit_width=bit_width,
        signed=signed,
        training=training)


def quant_output_scale_impl(fn, inp, quant_input_scale, quant_weight_scale):
    channel_dim = -1 if fn == F.linear else 1
    output_scale_shape = compute_channel_view_shape(inp, channel_dim=channel_dim)
    output_scale = quant_weight_scale.view(output_scale_shape)
    output_scale = output_scale * quant_input_scale.view(output_scale_shape)
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
