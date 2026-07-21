# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.autograd import Function

from brevitas.export.onnx.function import DynamoFn
from brevitas.nn.target.finn import PWPolyFEager

DOMAIN_STRING = "finn.pwpolyf"
DOMAIN_VERSION = 1


class FINNPWPolyFTorchScriptFn(Function):

    @staticmethod
    def symbolic(
            g, x, coeffs, func, K, degree, neg_clamp, pos_clamp, pos_passthrough):
        ret = g.op(
            f"{DOMAIN_STRING}::PWPolyF",
            x,
            func_s=func,
            K_i=int(K),
            degree_i=int(degree))
        ret.setType(x.type())
        return ret

    @staticmethod
    def forward(
            ctx, x, coeffs, func, K, degree, neg_clamp, pos_clamp, pos_passthrough):
        return PWPolyFEager.evaluate(
            x,
            coeffs,
            int(K),
            int(degree),
            float(neg_clamp),
            float(pos_clamp),
            bool(pos_passthrough))


class FINNPWPolyFDynamoFn(DynamoFn):

    @staticmethod
    def symbolic(x, coeffs, func, K, degree, neg_clamp, pos_clamp, pos_passthrough):
        return torch.onnx.ops.symbolic(
            f"{DOMAIN_STRING}::PWPolyF",
            (x,),
            {"func": func, "K": int(K), "degree": int(degree)},
            dtype=x.dtype,
            shape=x.shape,
            version=DOMAIN_VERSION)


class FINNPWPolyFOp:
    torchscript = FINNPWPolyFTorchScriptFn
    dynamo = FINNPWPolyFDynamoFn
