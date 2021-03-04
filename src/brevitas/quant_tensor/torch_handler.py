from warnings import warn

import torch.nn.functional as F
from brevitas.quant_tensor import QuantTensor


QUANT_TENSOR_FN_HANDLER = {}


def inplace_warn(inplace: bool):
    if inplace:
        warn("In-place semantics are ignored for QuantTensor, don't rely on them")


def quant_invariant_handler(fn, inp: QuantTensor):
    if inp.is_not_none:
        return inp.set(value=fn(inp.value))
    else:
        return fn(inp.value)


def relu_qt_handler(inp, inplace=False):
    inplace_warn(inplace)
    return quant_invariant_handler(F.relu, inp)


def relu6_qt_handler(inp, inplace=False):
    inplace_warn(inplace)
    return quant_invariant_handler(F.relu6, inp)


def hardtanh_qt_handler(inp, inplace=False):
    inplace_warn(inplace)
    return quant_invariant_handler(F.hardtanh, inp)




