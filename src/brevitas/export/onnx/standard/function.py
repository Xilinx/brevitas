# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.autograd import Function

from brevitas.export.onnx import onnx_export_opset

AXIS_OPSET = 13


class DequantizeLinearFn(Function):

    @staticmethod
    def symbolic(g, x, input_scale, input_zero_point, input_axis):
        opset_version = onnx_export_opset()

        if input_axis is not None and opset_version < AXIS_OPSET:
            raise RuntimeError('ONNX Opset 13 is required for per-channel quantization')
        elif input_axis is not None and opset_version >= AXIS_OPSET:
            ret = g.op('DequantizeLinear', x, input_scale, input_zero_point, axis_i=input_axis)
        else:
            ret = g.op('DequantizeLinear', x, input_scale, input_zero_point)
        return ret

    @staticmethod
    def forward(ctx, int_x, input_scale, input_zero_point, input_axis):
        if input_axis is not None:
            shape = [
                1,] * len(int_x.shape)
            shape[input_axis] = -1
            shape = tuple(shape)
        else:
            shape = (1,)
        return (int_x.float() -
                input_zero_point.float().reshape(shape)) * input_scale.reshape(shape)


class IntClipFn(Function):

    @staticmethod
    def symbolic(g, int_x, min_int_val, max_int_val):
        ret = g.op('Clip', int_x, min_int_val, max_int_val)
        return ret

    @staticmethod
    def forward(ctx, int_x, min_int_val, max_int_val):
        return torch.clamp(int_x, min_int_val, max_int_val)


class QuantizeLinearFn(Function):

    @staticmethod
    def symbolic(g, x, output_scale, ouput_zero_point, output_dtype, output_axis):
        opset_version = onnx_export_opset()

        if output_axis is not None and opset_version < AXIS_OPSET:
            raise RuntimeError('ONNX Opset 13 is required for per-channel quantization')
        elif output_axis is not None and opset_version >= AXIS_OPSET:
            ret = g.op('QuantizeLinear', x, output_scale, ouput_zero_point, axis_i=output_axis)
        else:
            ret = g.op('QuantizeLinear', x, output_scale, ouput_zero_point)
        return ret

    @staticmethod
    def forward(ctx, x, output_scale, output_zero_point, output_dtype, output_axis):
        if output_axis is not None:
            shape = [
                1,] * len(x.shape)
            shape[output_axis] = -1
            shape = tuple(shape)
        else:
            shape = (1,)
        int_x = ((x + output_zero_point.float().reshape(shape)) / output_scale.reshape(shape))
        if output_dtype == torch.uint8:
            min_int_val, max_int_val = 0., (2. ** 8) - 1
        elif output_dtype == torch.int8:
            min_int_val, max_int_val = -(2. ** 7), (2. ** 7) - 1
        elif output_dtype == torch.int32:
            min_int_val, max_int_val = -(2. ** 31), (2. ** 31) - 1
        else:
            raise RuntimeError(f"{output_dtype} not supported.")
        int_x = torch.clamp(int_x, min_int_val, max_int_val).to(output_dtype)
        return int_x
