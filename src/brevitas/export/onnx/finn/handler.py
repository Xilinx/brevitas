# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.nn.target.finn import PWPolyFActivation

from .function import FINNPWPolyFOp


class FINNPWPolyFHandler(ONNXBaseHandler):
    handled_layer = PWPolyFActivation

    def prepare_for_export(self, module: PWPolyFActivation):
        self.coeffs = module.coeffs
        self.func = module.func
        self.K = module.K
        self.degree = module.degree

    def symbolic_execution(self, x: Tensor):
        return self.export_op(FINNPWPolyFOp, x, self.coeffs, self.func, self.K, self.degree)
