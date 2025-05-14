# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This code was adapted from https://github.com/intel/auto-round, under the following LICENSE:
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gguf
import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
GGML_QUANT_SIZES = {
    gguf.GGMLQuantizationType.BF16: (1, 2),
    gguf.GGMLQuantizationType.Q4_0: (32, 2 + 16),
    gguf.GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
    gguf.GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),
    gguf.GGMLQuantizationType.Q8_0: (32, 2 + 32)}

GGML_QUANT_BLOCK = {}


def register_block(name):

    def register(cls):
        GGML_QUANT_BLOCK[name] = cls
        return cls

    return register


def ggml_quant(
        data: np.array, ggml_type, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    import torch
    data = data.squeeze().cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    scale = scale.detach().numpy() if isinstance(scale, torch.Tensor) else scale
    zp = zp.detach().numpy() if isinstance(zp, torch.Tensor) else zp
    wmin_m = wmin_m.detach().numpy() if isinstance(wmin_m, torch.Tensor) else wmin_m
    d_scale = d_scale.detach().numpy() if isinstance(d_scale, torch.Tensor) else d_scale
    d_wmin_m = d_wmin_m.detach().numpy() if isinstance(d_wmin_m, torch.Tensor) else d_wmin_m
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    # data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    quant_func = GGML_QUANT_BLOCK[ggml_type]
    if ggml_type == gguf.gguf.GGMLQuantizationType.Q4_K:
        new_data = quant_func(blocks, scale, zp, wmin_m=wmin_m, d_scale=d_scale, d_wmin_m=d_wmin_m)
    else:
        new_data = quant_func(blocks, scale, zp)

    assert new_data.dtype == np.uint8, "No uint8"
    assert new_data.shape[-1] == type_size, "No correct shape"
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return new_data


@register_block(gguf.GGMLQuantizationType.BF16)
def bf16_quant_block(blocks: np.array, scale=None, zp=None):
    n = blocks.view(np.uint32)
    # force nan to quiet
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16),
                 n)
    # round to nearest even
    n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype(np.uint16).view(np.uint8)


@register_block(gguf.GGMLQuantizationType.Q4_0)
def q4_0_quant_block(blocks: np.array, scale=None, zp=None):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0][0]
    blocks = (blocks.astype(np.float32) + np.float32(8)).astype(np.uint8).clip(0, 15)
    blocks = blocks.reshape((n_blocks, 2, block_size // 2))
    blocks = blocks[..., 0, :] | (blocks[..., 1, :] << np.uint8(4))
    d = d.astype(np.float16).view(np.uint8)

    return np.concatenate([d, blocks], axis=-1)


@register_block(gguf.GGMLQuantizationType.Q4_1)
def q4_1_quant_block(blocks: np.array, scale=None, zp=None):
    if scale is not None:
        d = scale.reshape((-1, 1))
        min = zp.reshape((-1, 1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 15
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_1][0]
    blocks = blocks.reshape((n_blocks, 2, block_size // 2))
    blocks = blocks[..., 0, :] | (blocks[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)
    m = min.astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, blocks], axis=-1)


@register_block(gguf.GGMLQuantizationType.Q8_0)
def q8_0_quant_block(blocks: np.array, scale=None, zp=None) -> np.ndarray:
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        d = abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    # (n_blocks, 2)
    d = d.astype(np.float16).view(np.uint8)
    # (n_blocks, block_size)
    blocks = blocks.astype(np.int8).view(np.uint8)

    return np.concatenate([d, blocks], axis=1)


@register_block(gguf.GGMLQuantizationType.Q4_K)
def q4_k_quant_block(
        blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    nb = blocks.shape[0]
    blocks = blocks.reshape(nb, QK_K // 32, 32)  # (nb, 8, 32)

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    output_qs = np.empty((nb, QK_K // 64, 32), dtype=np.uint8)

    scales = scale.reshape(-1, QK_K // 32)
    mins = wmin_m.reshape(-1, QK_K // 32)
    output_d = d_scale.reshape(-1, 1).astype(np.float32)
    output_dmin = d_wmin_m.reshape(-1, 1).astype(np.float32)
    inv_scale_scales = np.where(output_d == 0, 0, 1 / output_d)
    inv_scale_mins = np.where(output_dmin == 0, 0, 1 / output_dmin)

    # 6-bit quant for miniblock scales and zp
    q_scales = np.round(inv_scale_scales * scales).astype(np.uint8).clip(0, 63)
    q_mins = np.round(inv_scale_mins * mins).astype(np.uint8).clip(0, 63)

    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]

    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= ((q_scales[:, 4:] >> 4) << 6)
    output_scale[:, 4:8] |= ((q_mins[:, 4:] >> 4) << 6)

    output_qs = blocks[:, ::2] | (blocks[:, 1::2] << 4)

    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2)

    # [d, dmin, scale, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qs], axis=-1)
