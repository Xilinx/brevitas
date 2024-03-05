# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from enum import Enum

import pytest
import torch

from brevitas.inject.enum import QuantType
from brevitas.nn import QuantIdentity
from brevitas.quant_tensor import QuantTensor


class Operator(Enum):
    ADD = 0
    SUBTRACT = 1
    DIVIDE = 2
    MULTIPLY = 3
    MATMUL = 4


# QuantTensor isn't meant to be initialized directly, it'll be invalid if you do
# so you need to create it indirectly via QuantIdentity for example
def to_quant_tensor(input: torch.Tensor) -> QuantTensor:
    mod = QuantIdentity(bit_width=8, quant_type=QuantType.INT, return_quant_tensor=True)
    return mod(input)


def test_quant_tensor_init():
    x = torch.ones(4, 4)
    quant_tensor = QuantTensor(x)
    normal_tensor = torch.Tensor(x)

    assert torch.isclose(normal_tensor, quant_tensor, atol=0.1).all().item()


@pytest.mark.parametrize(
    'op', [Operator.ADD, Operator.SUBTRACT, Operator.DIVIDE, Operator.MULTIPLY, Operator.MATMUL])
def test_quant_tensor_operators(op):
    x = torch.ones(4, 4)

    a = torch.Tensor(x)
    b = torch.Tensor(x)
    qa = to_quant_tensor(a)
    qb = to_quant_tensor(b)

    if op == Operator.ADD:
        normal = a + b
        quant = qa + qb
    elif op == Operator.SUBTRACT:
        normal = a - b
        quant = qa - qb
    elif op == Operator.DIVIDE:
        normal = a / b
        quant = qa / qb
    elif op == Operator.MULTIPLY:
        normal = a * b
        quant = qa * qb
    elif op == Operator.MATMUL:
        normal = a @ b
        # @ matmul operator not implemented for QuantTensor
        quant = torch.matmul(qa, qb)
    else:
        # unrecognised operator
        assert False

    # tolerance set to a high value as there is considerable loss of precision
    assert torch.isclose(normal, quant, atol=0.1).all().item()


def test_quant_tensor_div_by_zero():
    a = to_quant_tensor(torch.ones(4, 4))
    b = to_quant_tensor(torch.zeros(4, 4))
    assert torch.isinf(a / b).all().item()


def test_quant_tensor_div_by_fraction():
    a = to_quant_tensor(torch.ones(4, 4))
    b = to_quant_tensor(torch.ones(4, 4) * 0.5)
    assert torch.isclose(a / b, torch.ones(4, 4) * 2, atol=0.1).all().item()


def test_quant_tensor_transpose():
    x = torch.ones(4, 4).tril()
    a = x.clone()
    b = to_quant_tensor(x)
    assert torch.isclose(a.transpose(0, 1), b.transpose(0, 1), atol=0.01).all().item()


def test_quant_tensor_view():
    x = torch.ones(4, 4)
    a = QuantTensor(x)
    b = torch.Tensor(x)

    assert torch.isclose(a.view(-1), b.view(-1), atol=0.01).all().item()
    assert torch.isclose(a.view(2, -1), b.view(2, -1), atol=0.01).all().item()
    assert torch.isclose(a.view(16, -1), b.view(16, -1), atol=0.01).all().item()
    assert torch.isclose(a.view(8, 2), b.view(8, 2), atol=0.01).all().item()


def test_is_valid():
    x = torch.randn(4, 4)
    # directly initialised QuantTensor shouldn't be valid
    invalid_quant_tensor = QuantTensor(x)
    assert invalid_quant_tensor.is_valid == False

    valid_quant_tensor = to_quant_tensor(x)
    assert valid_quant_tensor.is_valid


if __name__ == "__main__":
    test_quant_tensor_operators(Operator.MATMUL)
