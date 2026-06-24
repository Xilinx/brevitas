# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.autograd import Function

from brevitas.nn.target.finn import PWPolyFEager

from .custom_ops import DOMAIN_STRING


class FINNPWPolyFFn(Function):

    @staticmethod
    def symbolic(g, x, coeffs, func, K, degree):
        ret = g.op(
            f"{DOMAIN_STRING}::PWPolyF",
            x,
            func_s=func,
            K_i=int(K),
            degree_i=int(degree))
        ret.setType(x.type())
        return ret

    @staticmethod
    def forward(ctx, x, coeffs, func, K, degree):
        eager_impl = PWPolyFEager(func, int(K), int(degree))
        return eager_impl.evaluate(x, coeffs)
