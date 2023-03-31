# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import Optional, Tuple, Union
import warnings

from packaging import version
from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.quant_tensor import QuantTensor

DEFAULT_OPSET = 13


class StdONNXBaseManager(ONNXBaseManager, ABC):

    @classmethod
    def solve_onnx_opset(cls, export_kwargs):
        ka = 'opset_version'
        if ka not in export_kwargs:
            export_kwargs[ka] = DEFAULT_OPSET
            warnings.warn(f"ONNX opset version set to {DEFAULT_OPSET}, override with {ka}=")

    @classmethod
    def solve_enable_onnx_checker(cls, export_kwargs):
        ka = 'enable_onnx_checker'
        if (torch_version >= version.parse('1.5.0') and torch_version <= version.parse('1.10.0') and
                ka not in export_kwargs):
            export_kwargs[ka] = True

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            args: Optional[Union[Tensor, QuantTensor, Tuple]],
            export_path: Optional[str],
            input_shape: Optional[Tuple[int, ...]],
            input_t: Optional[Union[Tensor, QuantTensor]],
            disable_warnings,
            **onnx_export_kwargs):
        cls.solve_onnx_opset(onnx_export_kwargs)
        output = super().export_onnx(
            module, args, export_path, input_shape, input_t, disable_warnings, **onnx_export_kwargs)
        return output
