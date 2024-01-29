# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import onnx
import torch
from torch.autograd import Function

from brevitas.export.onnx import onnx_export_opset

AXIS_OPSET = 13

DATATYPE_DICT = {
    torch.float32: onnx.TensorProto.DataType.FLOAT,
    torch.float16: onnx.TensorProto.DataType.FLOAT16,
    torch.bfloat16: onnx.TensorProto.DataType.BFLOAT16}


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
        return int_x.float()


class IntClipFn(Function):

    @staticmethod
    def symbolic(g, int_x, min_int_val, max_int_val):
        ret = g.op('Clip', int_x, min_int_val, max_int_val)
        return ret

    @staticmethod
    def forward(ctx, int_x, min_int_val, max_int_val):
        return int_x


class CastFn(Function):

    @staticmethod
    def symbolic(g, x, dtype):
        ret = g.op('Cast', x, to_i=DATATYPE_DICT[dtype])
        return ret

    @staticmethod
    def forward(ctx, x, dtype):
        return x.to(dtype)


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
    def forward(ctx, x, output_scale, ouput_zero_point, output_dtype, output_axis):
        return x.type(output_dtype)


class DynamicQuantizeLinearFn(Function):

    @staticmethod
    def symbolic(g, x, output_dtype):
        x, scale, zp = g.op('DynamicQuantizeLinear', x, outputs=3)
        return x, scale, zp

    @staticmethod
    def forward(ctx, x, output_dtype):
        device = x.device
        dtype = x.dtype
        scale = torch.empty(1, device=device, dtype=dtype)
        zero_point = torch.empty(1, device=device, dtype=output_dtype)
        return x.type(output_dtype), scale, zero_point
