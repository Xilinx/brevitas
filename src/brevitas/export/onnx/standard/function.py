# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes

from brevitas.export.onnx import onnx_export_opset


class MatMulNBitsFn(Function):

    @staticmethod
    def symbolic(g, x, int_weights, scales, zero_points, K, N, bits, block_size):
        ret = g.op(
            'com.microsoft::MatMulNBits',
            x,
            int_weights,
            scales,
            zero_points,
            K_i=K,
            N_i=N,
            bits_i=bits,
            block_size_i=block_size)
        output_size = _get_tensor_sizes(x)
        output_size[-1] = N
        ret.setType(x.type().with_sizes(output_size))
        return ret

    @staticmethod
    def forward(g, x, int_weights, scales, zero_points, K, N, bits, block_size):
        dtype = x.dtype
        device = x.device
        shape = x.shape
        out_shape = list(shape)
        out_shape[-1] = N
        # Only tensor metadata (shape, dtype, device) are preserved in the forward pass during
        # tracing, not the correct value
        out = torch.empty(out_shape, dtype=dtype, device=device)
        return out


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
        return int_x.float()


class IntClipFn(Function):

    @staticmethod
    def symbolic(g, int_x, min_int_val, max_int_val):
        ret = g.op('Clip', int_x, min_int_val, max_int_val)
        return ret

    @staticmethod
    def forward(ctx, int_x, min_int_val, max_int_val):
        return int_x


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
