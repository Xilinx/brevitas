import functools

import torch
import torch.nn.functional as F

FLOAT_QUANT_TENSOR_FN_HANDLER = {}


def implements_float_qt(torch_function):

    @functools.wraps(torch_function)
    def decorator(func):
        FLOAT_QUANT_TENSOR_FN_HANDLER[torch_function] = func
        return func

    return decorator


@implements_float_qt(torch.cat)
def cat_handler(*args, **kwargs):
    from brevitas.quant_tensor import FloatQuantTensor
    return FloatQuantTensor.cat(*args, **kwargs)


@implements_float_qt(F.conv1d)
def conv1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.conv2d)
def conv2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.conv3d)
def conv3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.conv_transpose1d)
def conv_transpose1d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose1d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.conv_transpose2d)
def conv_transpose2d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose2d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.conv_transpose3d)
def conv_transpose3d_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.conv_transpose3d, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.linear)
def linear_handler(quant_input, quant_weight, bias=None, *args, **kwargs):
    output = quant_layer(F.linear, quant_input, quant_weight, bias, *args, **kwargs)
    return output


@implements_float_qt(F.embedding)
def embedding_handler(input, quant_weight, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor
    from brevitas.quant_tensor import FloatQuantTensor

    quant_weight_value = _unpack_quant_tensor(quant_weight)
    out = F.embedding(input, quant_weight_value, *args, **kwargs)

    if isinstance(quant_weight, FloatQuantTensor):
        scale = quant_weight.scale
        zero_point = quant_weight.zero_point
        exponent_bit_width = quant_weight.exponent_bit_width
        mantissa_bit_width = quant_weight.mantissa_bit_width
        exponent_bias = quant_weight.exponent_bias
        inf_values = quant_weight.inf_values
        nan_values = quant_weight.nan_values
        if any(t.numel() > 1 for t in [scale, zero_point]):
            raise RuntimeError("Only per-tensor quantization is supported.")
        signed = quant_weight.signed
        training = quant_weight.training
        saturating = quant_weight.saturating
        out = FloatQuantTensor(
            out,
            scale,
            zero_point,
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            signed,
            training)
    return out


@implements_float_qt(F.avg_pool2d)
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

    return x


@implements_float_qt(F.adaptive_avg_pool2d)
def adaptive_avg_pool2d_handler(quant_input, output_shape):
    from brevitas.quant_tensor import _unpack_quant_tensor

    x = F.adaptive_avg_pool2d(_unpack_quant_tensor(quant_input), output_shape)
    return x


def quant_layer(fn, quant_input, quant_weight, bias, *args, **kwargs):
    from brevitas.quant_tensor import _unpack_quant_tensor

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

    return output
