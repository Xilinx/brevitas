import functools

import torch
import torch.nn.functional as F


QUANT_TENSOR_FN_HANDLER = {}


def implements(torch_function):
    @functools.wraps(torch_function)
    def decorator(func):
        QUANT_TENSOR_FN_HANDLER[torch_function] = func
        return func
    return decorator


def quant_invariant_handler(fn, inp, *args, **kwargs):
    out_value = fn(inp.value, *args, **kwargs)
    if inp.is_not_none:
        return inp.set(value=out_value)
    else:
        return out_value


@implements(torch.flatten)
def flatten_handler(inp, *args, **kwargs):
    return inp.flatten(*args, **kwargs)


@implements(torch.reshape)
def reshape_handler(inp, *args, **kwargs):
    return inp.reshape(*args, **kwargs)


@implements(torch.transpose)
def transpose_handler(inp, *args, **kwargs):
    return inp.transpose(*args, **kwargs)


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

