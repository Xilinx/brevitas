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
import gguf.quants as _gguf_quants
import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
GROUP_MAX_EPS = 1e-30
GGML_QUANT_SIZES = {
    gguf.GGMLQuantizationType.BF16: (1, 2),
    gguf.GGMLQuantizationType.Q4_0: (32, 2 + 16),
    gguf.GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
    gguf.GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),
    gguf.GGMLQuantizationType.Q6_K: (256, QK_K // 2 + QK_K // 4 + QK_K // 16 + 2),
    gguf.GGMLQuantizationType.Q8_0: (32, 2 + 32)}


def _np_roundf(n: np.ndarray) -> np.ndarray:
    # round half away from zero, matching C nearest_int
    a = np.abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


def _make_qx_quants(x: np.ndarray, nmax: int, rmse_type: int = 1) -> np.ndarray:
    # vectorized port of ggml-quants.c:make_qx_quants; returns per-block scale
    amax_idx = np.abs(x).argmax(axis=-1, keepdims=True)
    max_val = np.take_along_axis(x, amax_idx, axis=-1).squeeze(-1).astype(np.float32)
    amax = np.abs(max_val)
    nonzero = amax >= GROUP_MAX_EPS
    safe_max = np.where(nonzero, max_val, np.float32(1.0))

    if rmse_type == 1:
        w = (x * x).astype(np.float32)
    elif rmse_type == 2:
        w = np.ones_like(x, dtype=np.float32)
    elif rmse_type == 3:
        w = np.abs(x).astype(np.float32)
    else:
        w = np.sqrt(np.abs(x)).astype(np.float32)

    def _eval(iscale_):
        L = _np_roundf(iscale_[..., None] * x).clip(-nmax, nmax - 1).astype(np.float32)
        sumlx = (w * x * L).sum(axis=-1)
        suml2 = (w * L * L).sum(axis=-1)
        return sumlx, suml2

    iscale0 = np.where(nonzero, -np.float32(nmax) / safe_max, np.float32(0.0))
    sumlx, suml2 = _eval(iscale0)
    safe_l2 = np.where(suml2 != 0, suml2, np.float32(1.0))
    scale = np.where(suml2 != 0, sumlx / safe_l2, np.float32(0.0))
    best = scale * sumlx

    for is_val in range(-9, 10):
        if is_val == 0:
            continue
        iscale_try = np.where(
            nonzero, -(np.float32(nmax) + np.float32(0.1) * is_val) / safe_max, np.float32(0.0))
        sumlx_t, suml2_t = _eval(iscale_try)
        better = (suml2_t > 0) & (sumlx_t * sumlx_t > best * suml2_t)
        new_scale = np.where(
            suml2_t != 0,
            sumlx_t / np.where(suml2_t != 0, suml2_t, np.float32(1.0)),
            np.float32(0.0))
        scale = np.where(better, new_scale, scale)
        best = np.where(better, new_scale * sumlx_t, best)

    return np.where(nonzero, scale, np.float32(0.0)).astype(np.float32)


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

    if scale.dtype not in (torch.float16, torch.float32):
        scale = scale.to(torch.float32)
    scale = scale.detach().numpy() if isinstance(scale, torch.Tensor) else scale

    if zp.dtype not in (torch.float16, torch.float32):
        zp = zp.to(torch.float32)
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


@register_block(gguf.GGMLQuantizationType.Q6_K)
def q6_k_quant_block(blocks: np.array, scale=None, zp=None):
    # port of ggml-quants.c:quantize_row_q6_K_ref; always self-derives scales
    nb = blocks.shape[0]
    sub = blocks.reshape(nb, QK_K // 16, 16).astype(np.float32)
    sub_scales = _make_qx_quants(sub, nmax=32, rmse_type=1)

    abs_scales = np.abs(sub_scales)
    max_abs = abs_scales.max(axis=-1)
    nonzero = max_abs >= GROUP_MAX_EPS
    imax = abs_scales.argmax(axis=-1, keepdims=True)
    max_scale = np.take_along_axis(sub_scales, imax, axis=-1).squeeze(-1)
    safe_max_scale = np.where(max_scale != 0, max_scale, np.float32(1.0))
    iscale = np.where(nonzero, np.float32(-128.0) / safe_max_scale, np.float32(0.0))
    safe_iscale = np.where(iscale != 0, iscale, np.float32(1.0))
    d = np.where(nonzero, np.float32(1.0) / safe_iscale, np.float32(0.0))

    # 8-bit quant for sub-block scales
    q_scales = np.clip(_np_roundf(iscale[:, None] * sub_scales), -128, 127).astype(np.int8)
    q_scales = np.where(nonzero[:, None], q_scales, np.int8(0))

    # 6-bit quant per element using d * sub_scale
    d_eff = d[:, None].astype(np.float32) * q_scales.astype(np.float32)
    safe_d_eff = np.where(d_eff != 0, d_eff, np.float32(1.0))
    inv_d_eff = np.where(d_eff != 0, np.float32(1.0) / safe_d_eff, np.float32(0.0))
    L = _np_roundf(sub * inv_d_eff[:, :, None]).clip(-32, 31).astype(np.int32) + 32
    L = np.where(nonzero[:, None, None], L, 0).astype(np.uint8).reshape(nb, QK_K)

    # Pack into ql (128B) and qh (64B), interleaving four 32-elem groups per half-block.
    # See ggml-quants.c:quantize_row_q6_K_ref for the bit layout.
    ql = np.empty((nb, QK_K // 2), dtype=np.uint8)
    qh = np.empty((nb, QK_K // 4), dtype=np.uint8)
    for half in range(2):
        j = half * 128
        a = L[:, j:j + 32]
        b = L[:, j + 32:j + 64]
        c = L[:, j + 64:j + 96]
        e = L[:, j + 96:j + 128]
        ql[:, half * 64:half * 64 + 32] = (a & 0x0F) | ((c & 0x0F) << 4)
        ql[:, half * 64 + 32:half * 64 + 64] = (b & 0x0F) | ((e & 0x0F) << 4)
        qh[:, half * 32:half * 32 + 32] = ((a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) |
                                           ((e >> 4) << 6))

    scales_bytes = q_scales.view(np.uint8)
    d_for_pack = np.where(nonzero, d, np.float32(0.0))
    d_bytes = d_for_pack.astype(np.float16).reshape(nb, 1).view(np.uint8)

    # [ql, qh, scales, d]
    return np.concatenate([ql, qh, scales_bytes, d_bytes], axis=-1)


# Route gguf.quants.quantize(data, Q6_K) through our encoder; gguf-py ships
# only the K-family dequantizer, so without this the convert.py fallback
# silently regresses Q6_K targets back to F32.
_gguf_quants.Q6_K.quantize_blocks = classmethod(
    lambda cls, blocks: q6_k_quant_block(blocks, scale=None, zp=None))
