# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
import torch
from torch.nn import BatchNorm3d
from torch.nn import Conv3d

from brevitas.inject.defaults import Int8BiasPerTensorFloatInternalScaling
from brevitas.nn import QuantConv3d
from brevitas.nn.utils import merge_bn

OUTPUT_CHANNELS = 10
INPUT_CHANNELS = 5
KERNEL_SIZE = (3, 3, 3)
WEIGHT_BIT_WIDTH = 5


class TestQuantConv3d:

    @pytest_cases.parametrize(
        'kwargs',
        [{}, {
            'padding': 'same', 'stride': 1}, {
                'padding': 'same', 'stride': (1, 1, 1)}, {
                    'padding': 'same', 'stride': (2, 1, 1)}],
        ids=[
            'defaults',
            'padding="same",stride=1',
            'padding="same",stride=(1,1,1)',
            'padding="same",stride=(2,1,1)'])
    def test_module_init(self, kwargs):
        mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False,
            **kwargs)

    def test_fp_quant_module(self):
        float_mod = Conv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_type='FP',
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_none_weight_quant_module(self):
        float_mod = Conv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant=None,
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_delayed_quant_module(self):
        float_mod = Conv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_internally_scaled_int_bias(self):
        mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=True,
            bias_quant=Int8BiasPerTensorFloatInternalScaling)
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20, 20)
        mod(inp)

    def test_internally_scaled_int_bias_after_bn_merge(self):
        mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=False,
            bias_quant=Int8BiasPerTensorFloatInternalScaling)
        bn = BatchNorm3d(OUTPUT_CHANNELS)
        merge_bn(mod, bn)
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20, 20)
        mod(inp)

    @pytest_cases.parametrize(
        'kwargs',
        [{
            'is_same_padded_strided': False}, {
                'padding': 'same', 'stride': 1, 'is_same_padded_strided': False}, {
                    'padding': 'same', 'stride': (1, 1, 1), 'is_same_padded_strided': False}, {
                        'padding': 'same', 'stride': (2, 1, 1), 'is_same_padded_strided': True}, {
                            'padding': 0, 'stride': (2, 1, 1), 'is_same_padded_strided': False}],
        ids=[
            'defaults',
            'padding="same",stride=1',
            'padding="same",stride=(1,1,1)',
            'padding="same",stride=(2,1,1)',
            'padding=0,stride=(2,1,1)'])
    def test_is_same_padded_strided(self, kwargs):
        is_same_padded_strided = kwargs['is_same_padded_strided']
        del kwargs['is_same_padded_strided']
        mod = QuantConv3d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False,
            **kwargs)
        assert is_same_padded_strided == mod.is_same_padded_strided, f"Expected is_same_padded_strided={is_same_padded_strided}, found is_same_padded_strided={mod.is_same_padded_strided}"
