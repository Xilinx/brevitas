# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import onnxscript
from onnxscript import FLOAT
import torch

import brevitas.library
from brevitas.nn.target.finn import pwpolyf_eager_from_attrs

__all__ = [
    "DOMAIN_STRING",
    "DOMAIN_VERSION",
    "LIBRARY_STRING",
    "PWPolyF",
    "pwpolyf",
    "pwpolyf_wrapper",]

LIBRARY_STRING = "finn"
DOMAIN_STRING = "finn.pwpolyf"
DOMAIN_VERSION = 1
finn_op = onnxscript.values.Opset(domain=DOMAIN_STRING, version=DOMAIN_VERSION)


@brevitas.library.custom_op(f"{LIBRARY_STRING}::pwpolyf", mutates_args=())
def pwpolyf(x: torch.Tensor, func: str, K: int, degree: int) -> torch.Tensor:
    return pwpolyf_eager_from_attrs(x, func, K, degree)


@pwpolyf.register_fake
def _pwpolyf_fake(tensor_x, func, K, degree):
    return torch.empty_like(tensor_x)


@onnxscript.script(finn_op, default_opset=finn_op)
def PWPolyF(x: FLOAT, func: str, K: int, degree: int) -> FLOAT:
    return x


@onnxscript.script(finn_op, default_opset=finn_op)
def pwpolyf_wrapper(x: FLOAT, func: str, K: int, degree: int) -> FLOAT:
    return PWPolyF(x, func, K, degree)
