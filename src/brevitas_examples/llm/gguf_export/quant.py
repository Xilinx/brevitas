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
from gguf import GGML_QUANT_SIZES
from gguf import QK_K
import gguf.quants as _gguf_quants
import numpy as np
import torch

GROUP_MAX_EPS = 1e-30


def _make_qx_quants(x: np.ndarray, nmax: int) -> np.ndarray:
    """Per-block symmetric scale search (ggml-quants.c:make_qx_quants, rmse_type=1).

    For every block (the last axis of ``x``), return the scale ``s`` that minimizes
    the weighted reconstruction error of ``x ~= s * round(x / s)`` with importance
    weights ``w = x**2``. All-zero blocks get scale 0. Used for Q6_K sub-blocks.
    """
    # Anchor on the largest-magnitude element; its sign sets the scale sign.
    amax_idx = np.abs(x).argmax(axis=-1, keepdims=True)
    max_val = np.take_along_axis(x, amax_idx, axis=-1).squeeze(-1).astype(np.float32)
    nonzero = np.abs(max_val) >= GROUP_MAX_EPS
    safe_max = np.where(nonzero, max_val, np.float32(1.0))
    w = (x * x).astype(np.float32)  # importance weights (rmse_type=1)

    def _fit(iscale):
        # Quantize x to fixed integer codes L with the trial inverse-scale, then
        # solve for the continuous scale s that minimizes the weighted error
        #   E(s) = sum_i w_i (x_i - s * L_i)^2.
        # dE/ds = 0 gives the closed-form least-squares optimum s = sumlx / suml2,
        # so the residual (x - s*L) is never formed explicitly. Return the two
        # sufficient statistics; the resulting error reduction is sumlx^2 / suml2.
        L = _gguf_quants.np_roundf(iscale[..., None] * x).clip(-nmax, nmax - 1).astype(np.float32)
        sumlx = (w * x * L).sum(axis=-1)  # sum w * x * L
        suml2 = (w * L * L).sum(axis=-1)  # sum w * L^2
        return sumlx, suml2

    # Step 1: initial guess maps the max-magnitude element to the range edge.
    sumlx, suml2 = _fit(np.where(nonzero, -np.float32(nmax) / safe_max, np.float32(0.0)))
    scale = np.where(suml2 != 0, sumlx / np.where(suml2 != 0, suml2, 1.0), np.float32(0.0))
    best = scale * sumlx  # error reduction sumlx^2/suml2; maximizing it minimizes E

    # Step 2: try a small grid; keep the one with the largest error reduction per block.
    #  The test sumlx^2 > best*suml2 is that comparison cross-multiplied to avoid a division.
    for is_val in range(-9, 10):
        if is_val == 0:
            continue
        iscale = np.where(
            nonzero, -(np.float32(nmax) + np.float32(0.1) * is_val) / safe_max, np.float32(0.0))
        sumlx, suml2 = _fit(iscale)
        better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        new_scale = np.where(suml2 != 0, sumlx / np.where(suml2 != 0, suml2, 1.0), np.float32(0.0))
        scale = np.where(better, new_scale, scale)
        best = np.where(better, new_scale * sumlx, best)

    return np.where(nonzero, scale, np.float32(0.0)).astype(np.float32)


GGML_QUANT_BLOCK = {}


def register_block(name):

    def register(cls):
        GGML_QUANT_BLOCK[name] = cls
        return cls

    return register


def ggml_quant(
        data: np.array, ggml_type, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):

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

    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    quant_func = GGML_QUANT_BLOCK[ggml_type]
    if ggml_type == gguf.GGMLQuantizationType.Q4_K:
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
    # Pack pre-quantized signed codes in [-8, 7] with the given fp16 scale d.
    # gguf's native Q4_0 handles raw floats
    assert scale is not None
    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0][0]
    d = scale.reshape((-1, 1))
    q = (blocks.astype(np.float32) + np.float32(8)).clip(0, 15).astype(np.uint8)
    q = q.reshape((n_blocks, 2, block_size // 2))
    q = q[..., 0, :] | (q[..., 1, :] << np.uint8(4))
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, q], axis=-1)


@register_block(gguf.GGMLQuantizationType.Q4_1)
def q4_1_quant_block(blocks: np.array, scale=None, zp=None):
    # Pack pre-quantized codes in [0, 15] with scale d and zero-point zp; q4_1
    # stores the offset as min = -zp * d. gguf's native Q4_1 handles raw floats.
    assert scale is not None and zp is not None
    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_1][0]
    d = scale.reshape((-1, 1))
    m = zp.reshape((-1, 1)) * d * -1
    q = blocks.astype(np.float32).clip(0, 15).astype(np.uint8)
    q = q.reshape((n_blocks, 2, block_size // 2))
    q = q[..., 0, :] | (q[..., 1, :] << np.uint8(4))
    d = d.astype(np.float16).view(np.uint8)
    m = m.astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, q], axis=-1)


@register_block(gguf.GGMLQuantizationType.Q8_0)
def q8_0_quant_block(blocks: np.array, scale=None, zp=None) -> np.ndarray:
    # Pack pre-quantized int8 codes with the given fp16 scale d.
    # gguf's native Q8_0 handles raw floats
    assert scale is not None
    d = scale.reshape((-1, 1)).astype(np.float16).view(np.uint8)
    q = blocks.astype(np.int8).view(np.uint8)
    return np.concatenate([d, q], axis=1)


def _q4_k_pack(q_scales, q_mins, output_d, output_dmin, codes):
    # q_scales/q_mins: (nb, 8) uint8 6-bit; output_d/output_dmin: (nb, 1) float32;
    # codes: (nb, 8, 32) uint8 4-bit. Packs to the block_q4_K byte layout
    # (the get_scale_min_k4 6-bit interleave + nibble-packed quants).
    nb = codes.shape[0]
    output_scale = np.empty((nb, _gguf_quants.Q4_K.K_SCALE_SIZE), dtype=np.uint8)
    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]
    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= ((q_scales[:, 4:] >> 4) << 6)
    output_scale[:, 4:8] |= ((q_mins[:, 4:] >> 4) << 6)

    output_qs = (codes[:, ::2] | (codes[:, 1::2] << 4)).reshape(nb, QK_K // 2)
    d_bytes = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    dmin_bytes = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)

    # [d, dmin, scale, qs]
    return np.concatenate([d_bytes, dmin_bytes, output_scale, output_qs], axis=-1)


@register_block(gguf.GGMLQuantizationType.Q4_K)
def q4_k_quant_block(
        blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    # Pack pre-quantized codes in [0, 15] with the 8 sub-block scales/mins and their
    # fp16 super-scales d_scale/d_wmin_m.
    assert scale is not None and wmin_m is not None and d_scale is not None and d_wmin_m is not None
    nb = blocks.shape[0]
    scales = scale.reshape(-1, QK_K // 32)
    mins = wmin_m.reshape(-1, QK_K // 32)
    output_d = d_scale.reshape(-1, 1).astype(np.float32)
    output_dmin = d_wmin_m.reshape(-1, 1).astype(np.float32)
    inv_scale_scales = np.where(output_d == 0, 0, 1 / output_d)
    inv_scale_mins = np.where(output_dmin == 0, 0, 1 / output_dmin)
    q_scales = np.round(inv_scale_scales * scales).astype(np.uint8).clip(0, 63)
    q_mins = np.round(inv_scale_mins * mins).astype(np.uint8).clip(0, 63)
    codes = blocks.reshape(nb, QK_K // 32, 32).astype(np.uint8)
    return _q4_k_pack(q_scales, q_mins, output_d, output_dmin, codes)


def _q6_k_quantize_scales(sub_scales: np.ndarray):
    # Quantize the 16 per-sub-block scales to the Q6_K format: an fp16 super-block
    # scale d plus 16 int8 codes, with the max-magnitude scale anchored at -128.
    abs_scales = np.abs(sub_scales)
    nonzero = abs_scales.max(axis=-1) >= GROUP_MAX_EPS
    imax = abs_scales.argmax(axis=-1, keepdims=True)
    max_scale = np.take_along_axis(sub_scales, imax, axis=-1).squeeze(-1)
    safe_max_scale = np.where(max_scale != 0, max_scale, np.float32(1.0))
    iscale = np.where(nonzero, np.float32(-128.0) / safe_max_scale, np.float32(0.0))
    safe_iscale = np.where(iscale != 0, iscale, np.float32(1.0))
    d = np.where(nonzero, np.float32(1.0) / safe_iscale, np.float32(0.0))
    q_scales = np.clip(_gguf_quants.np_roundf(iscale[:, None] * sub_scales), -128,
                       127).astype(np.int8)
    q_scales = np.where(nonzero[:, None], q_scales, np.int8(0))
    return d, q_scales, nonzero


def _q6_k_pack(L: np.ndarray, q_scales: np.ndarray, d: np.ndarray, nonzero: np.ndarray):
    # Pack codes L (uint8, [0,63]) into ql (128B) and qh (64B), interleaving four
    # 32-elem groups per 128-elem half-block, then the int8 scales and fp16 d.
    # See ggml-quants.c:quantize_row_q6_K_ref for the bit layout.
    nb = L.shape[0]
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


@register_block(gguf.GGMLQuantizationType.Q6_K)
def q6_k_quant_block(blocks: np.array, scale=None, zp=None):
    # Adaptation of ggml-quants.c:quantize_row_q6_K_ref.
    #   scale is None -> blocks are raw floats; derive the 16 sub-block scales.
    #   scale given   -> blocks are pre-quantized codes in [-32, 31] and scale holds
    #                    the 16 per-sub-block scales (flattenable to (nb, QK_K/16)).
    nb = blocks.shape[0]
    if scale is None:
        sub = blocks.reshape(nb, QK_K // 16, 16).astype(np.float32)
        # Step 1: per-sub-block symmetric scale (16 sub-blocks of 16).
        sub_scales = _make_qx_quants(sub, nmax=32)
        # Step 2: quantize the 16 scales to int8 + an fp16 super-block scale d.
        d, q_scales, nonzero = _q6_k_quantize_scales(sub_scales)
        # Step 3: recompute the 6-bit codes from the quantized scale d * q_scales
        # (stored as L + 32 in [0, 63]), matching llama.cpp's two-pass quantization.
        d_eff = d[:, None].astype(np.float32) * q_scales.astype(np.float32)
        inv_d_eff = np.where(d_eff != 0, np.float32(1.0) / np.where(d_eff != 0, d_eff, 1.0), 0.0)
        L = _gguf_quants.np_roundf(sub * inv_d_eff[:, :, None]).clip(-32, 31).astype(np.int32) + 32
        L = np.where(nonzero[:, None, None], L, 0).astype(np.uint8).reshape(nb, QK_K)
    else:
        sub_scales = scale.reshape(nb, QK_K // 16).astype(np.float32)
        d, q_scales, nonzero = _q6_k_quantize_scales(sub_scales)
        L = (blocks.astype(np.int32) + 32).clip(0, 63).astype(np.uint8).reshape(nb, QK_K)

    return _q6_k_pack(L, q_scales, d, nonzero)


# gguf ships only the K-family dequantizer, so route gguf.quants.quantize(data, Q6_K)
# through our encoder; without this the convert.py pass-through path (the token_embd /
# output bump) would regress Q6_K targets back to F32.
_gguf_quants.Q6_K.quantize_blocks = classmethod(
    lambda cls, blocks: q6_k_quant_block(blocks, scale=None))
