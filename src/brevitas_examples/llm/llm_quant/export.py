"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
import warnings

import numpy as np
import torch
from torch.nn import Module
from torch.onnx import register_custom_op_symbolic

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.standard.function import MatMulNBitsFn
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.nn import QuantLinear
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


# TODO: Improve Groupwise export
def clip_kwargs(narrow, signed, bit_width):
    if narrow or bit_width != 8. and bit_width != 32.:
        if signed and (bit_width < 8. or narrow and bit_width <= 8.):
            dtype = torch.int8
        elif not signed and (bit_width < 8. or narrow and bit_width <= 8.):
            dtype = torch.uint8
        elif signed and (bit_width < 32. or narrow and bit_width <= 32.):
            dtype = torch.int32
        else:
            raise RuntimeError(f"Sign {signed} and bit width {bit_width} not supported for export.")
        return {
            'min_val': min_int(signed, narrow, bit_width).to(dtype),
            'max_val': max_int(signed, narrow, bit_width).to(dtype)}
    else:
        return None


class WeightBlockQuantHandlerBase(BaseHandler, ABC):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super(WeightBlockQuantHandlerBase, self).__init__()
        self.int_weight = None
        self.scale = None
        self.zero_point = None
        self.bit_width = None
        self.dtype = None

    @abstractmethod
    def prepare_for_export(self, module):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class WeightBlockQuantProxyHandler(WeightBlockQuantHandlerBase):
    handled_layer = GroupwiseWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.expanded_groupwise_shape = None
        self.reshaped_groupwise_shape = None
        self.expanded_zero_point_shape = None
        self.reshaped_zero_point_shape = None

    def prepare_for_export(self, module):
        assert len(module.tracked_module_list) == 1, "Shared quantizers not supported."
        quant_layer = module.tracked_module_list[0]
        quant_weight = quant_layer.quant_weight()
        self.bit_width = quant_weight.bit_width
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        signed = module.is_signed
        self.int_dtype = torch.int8 if signed else torch.uint8
        self.dtype = quant_weight.value.dtype
        self.scale = quant_weight.scale_
        self.expanded_scaling_shape = quant_weight.value_.shape
        self.reshaped_scaling_shape = quant_weight.value.shape
        if (quant_weight.zero_point != 0.).any():
            self.zero_point = quant_weight.zero_point_
        else:
            self.zero_point = None

        self.clip_kwargs = clip_kwargs(
            module.is_narrow_range, module.is_signed, quant_weight.bit_width)

    def forward(self, x):
        scale = self.scale
        zero_point = self.zero_point
        bit_width = self.bit_width
        # If zero point is not defined, it's all zeros
        if self.zero_point is None:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = self.zero_point

        # QCDQ
        x = x.view(self.expanded_groupwise_shape)
        x = torch.round((x / scale) + zero_point).type(self.int_dtype)
        if self.clip_kwargs is not None:
            x = torch.clip(x, min=self.clip_kwargs['min_val'], max=self.clip_kwargs['max_val'])
        x = (x.type(self.dtype) - zero_point) * scale

        # Fix shape post quantization
        # If zero_point is not defined, propagate same shape as scale
        if self.zero_point is None:
            zero_point = torch.zeros_like(scale).type(self.int_dtype)

        return x, scale, zero_point, bit_width


class LinearWeightBlockQuantHandler(WeightBlockQuantHandlerBase, ABC):
    handled_layer = QuantLinear

    def __init__(self):
        super(LinearWeightBlockQuantHandler, self).__init__()
        self.group_size = None

    def pack_int_weights(self, bit_width, int_weights):
        assert int_weights.dtype in [torch.int8, torch.uint8], "Packing requires (u)int8 input."
        if bit_width == 8:
            return int_weights
        elif bit_width == 4 or bit_width == 2:
            packed_int_weights = torch.zeros(
                (int_weights.shape[0], int_weights.shape[1] * bit_width // 8),
                device=int_weights.device,
                dtype=int_weights.dtype)
            i = 0
            for column in range(packed_int_weights.shape[1]):
                # Compared to the reference below we don't transpose the matrix and we pack into 8b data rather than 32b
                # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/05781593c818d4dc8adc2d32c975e83d17d2b9a8/quant/quant_linear.py#L346
                for j in range(i, i + (8 // bit_width)):
                    shift_factor = (bit_width * (j - i))
                    packed_int_weights[:, column] |= int_weights[:, j] << shift_factor
                i += 8 // bit_width
            return packed_int_weights

        # pack 3b values into 3 bytes, 5b values into 5 bytes, 6b values into 4 bytes
        elif bit_width == 3 or bit_width == 5 or bit_width == 6:
            padding = (int_weights.shape[1] * bit_width) % 8
            if padding > 0:
                warnings.warn(
                    f"Weight tensor does not divide by {bit_width}, zero-padding columns by {padding}."
                )
            packed_int_weights = torch.zeros(
                (int_weights.shape[0], (int_weights.shape[1] * bit_width + padding) // 8),
                device=int_weights.device,
                dtype=int_weights.dtype)

            def lcm(x, y):
                from fractions import gcd
                return x * y // gcd(x, y)

            num_packed_bits = lcm(bit_width, 8)
            num_packed_bytes = num_packed_bits // 8
            num_packed_elems = num_packed_bits // bit_width

            i = 0
            for column in range(0, packed_int_weights.shape[1], num_packed_bytes):
                # cast to uint8 since it's the only dtype supported by unpackbits
                # the bit-wise representation of int8 values isn't affected
                bits_to_unpack = int_weights[:, i:i + num_packed_elems].numpy().astype(np.uint8)
                unpacked_bits = np.unpackbits(bits_to_unpack, axis=1)
                unpacked_bits = unpacked_bits.reshape(unpacked_bits.shape[0], -1, 8)
                unpacked_bits = unpacked_bits[:, :, -bit_width:]
                unpacked_bits = unpacked_bits.reshape(unpacked_bits.shape[0], -1)
                packed_bits = np.packbits(unpacked_bits, axis=1)
                packed_int_weights[:, column:column +
                                   num_packed_bytes] |= torch.from_numpy(packed_bits)
                i += num_packed_elems
            return packed_int_weights
        else:
            raise ValueError(f"Bit width {bit_width} not supported.")

    def prepare_for_export(self, module):
        quant_weight = module.quant_weight()
        self.bit_width = quant_weight.bit_width
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        self.bias = module.bias
        self.scale = quant_weight.scale_
        if (quant_weight.zero_point != 0.).any():
            self.zero_point = quant_weight.zero_point_
        else:
            # if there is no zero-point, export zeroes in the shape of scale
            self.zero_point = torch.zeros_like(self.scale)
        self.group_size = quant_weight.group_size
        self.bit_width = int(self.bit_width.cpu().item())
        self.int_weight = self.pack_int_weights(self.bit_width, quant_weight.int().detach())

    @abstractmethod
    def forward(self, x):
        pass


class BlockQuantProxyLevelManager(BaseManager):

    handlers = [WeightBlockQuantProxyHandler]

    @classmethod
    def set_export_handler(cls, module):
        _set_proxy_export_handler(cls, module)


def block_quant_layer_level_manager(export_handlers, target=None, custom_fns_to_register=None):

    class BlockQuantLayerLevelManager(BaseManager):
        handlers = export_handlers
        target_name = '' if target is None else target
        custom_fns = [] if custom_fns_to_register is None else custom_fns_to_register

        @classmethod
        def set_export_handler(cls, module):
            _set_layer_export_handler(cls, module)

    return BlockQuantLayerLevelManager


@contextmanager
def brevitas_proxy_export_mode(model, export_manager):
    is_training = model.training
    model.eval()
    model.apply(export_manager.set_export_handler)
    _set_proxy_export_mode(model, enabled=True)
    try:
        yield model
    finally:
        _set_proxy_export_mode(model, enabled=False)
        model.train(is_training)


@contextmanager
def brevitas_layer_export_mode(model, export_manager):
    is_training = model.training
    model.eval()
    model.apply(export_manager.set_export_handler)
    _set_layer_export_mode(model, enabled=True)
    try:
        yield model
    finally:
        _set_layer_export_mode(model, enabled=False)
        model.train(is_training)


def replace_call_fn_target(graph_model, src, target):
    for node in graph_model.graph.nodes:
        if node.op == "call_function" and node.target is src:
            node.target = target
    graph_model.graph.lint()
    graph_model.recompile()


class ONNXLinearWeightBlockQuantHandlerFwd(ONNXBaseHandler, WeightBlockQuantHandlerBase):
    handled_layer = QuantLinear

    def __init__(self):
        super(ONNXLinearWeightBlockQuantHandlerFwd, self).__init__()
        self.group_size = None

    def pack_int_weights(self, bit_width, int_weights, zero_point):
        assert int_weights.dtype in [torch.uint8, torch.int8], "Packing requires (u)int8 input."
        assert bit_width == 4, "Only 4 bit quantization export is supported at the moment"

        is_symmetric = torch.sum(zero_point) == 0
        zero_point = zero_point.to(torch.uint8)
        rows, cols = int_weights.shape
        group_size = self.group_size
        blob_size = group_size // 2
        k_blocks = (rows + group_size - 1) // group_size
        padded_rows = k_blocks * group_size
        pad_len = padded_rows - rows

        # ONNX operator assumes implicit zp of 8 (largest negative number in Po2)
        # If we are in a "symmetric"  quantized scenario, we need to add this implicit zero point
        # Otherwise it has already been added during the convesion to integer.
        # This allows to pack weights always in unsigned integer.
        zp = 0 if not int_weights.dtype == torch.int8 else 8
        int_weights += zp
        if pad_len > 0:
            int_weights = torch.nn.functional(int_weights, (0, 0, 0, pad_len))
        packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
        rows, cols = int_weights.shape
        int_weights = int_weights.t()
        for n in range(cols):
            for k_id in range(0, rows, group_size):
                blk_int0 = (int_weights[n, k_id:k_id + group_size:2].numpy()).astype("uint8")
                blk_int1 = (int_weights[n, k_id + 1:k_id + group_size:2].numpy()).astype("uint8")
                packed[n, k_id // group_size] = np.bitwise_or(blk_int0, np.left_shift(blk_int1, 4))

        zero_point = zero_point.to(torch.uint8).flatten()

        # The constant value 136 is derived from the source code in ORT test suite.
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/python/quantization/test_quantizeblockwise_4bits.py
        base_zp = 136 if is_symmetric else 0
        packed_zp = base_zp * torch.ones(
            (zero_point.shape[0] + 1) // 2, device=int_weights.device, dtype=torch.uint8)

        i = 0
        for column in range(packed_zp.shape[0]):
            for j in range(i, i + (8 // bit_width)):
                shift_factor = (bit_width * (j - i))
                packed_zp[column] |= zero_point[j] << shift_factor
            i += 8 // bit_width
        return torch.tensor(packed), packed_zp

    def prepare_for_export(self, module):
        quant_weight = module.quant_weight()
        self.bit_width = quant_weight.bit_width
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        self.bias = module.bias
        self.scale = quant_weight.scale_
        if (quant_weight.zero_point != 0.).any():
            self.zero_point = quant_weight.zero_point_
        else:
            # if there is no zero-point, export zeroes in the shape of scale
            self.zero_point = torch.zeros_like(self.scale)
        self.group_size = module.weight_quant.quant_injector.group_size
        self.bit_width = int(self.bit_width.cpu().item())
        self.int_weight, self.zero_point = self.pack_int_weights(self.bit_width, quant_weight.int().t().detach(), self.zero_point)
        self.weight_shape = module.weight.shape

    def symbolic_execution(self, x):
        int_weights = self.int_weight
        scale = self.scale
        bit_width = self.bit_width
        N, K = self.weight_shape
        out = MatMulNBitsFn.apply(
            x, int_weights, scale.flatten(), self.zero_point, K, N, bit_width, self.group_size)
        return out


def export_packed_onnx(model, input, export_path):
    export_class = block_quant_layer_level_manager(
        export_handlers=[ONNXLinearWeightBlockQuantHandlerFwd],
        target='',
        custom_fns_to_register=MatMulNBitsFn)

    with torch.inference_mode(), brevitas_layer_export_mode(model, export_class):
        torch.onnx.export(model, input, export_path)
