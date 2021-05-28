# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Based on https://github.com/pytorch/pytorch/blob/
#            439930c81b5c711f30e739e900db7a0599659ead/torch/overrides.py

"""
Python implementation of __torch_function__

While most of the torch API and handling for __torch_function__ happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for __torch_function__ overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

NOTE: heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""


import functools
from typing import Dict, Callable
from packaging import version

import torch

IS_ABOVE_110 = version.parse(torch.__version__) > version.parse("1.1.0")


def get_tensor_overrides() -> Dict[Callable, Callable]:
    Tensor = torch.Tensor
    ret = {
        Tensor.__floordiv__: lambda self, other: -1,
        Tensor.__rfloordiv__: lambda self, other: -1,
        Tensor.__truediv__: lambda self, other: -1,
        Tensor.__itruediv__: lambda self, other: -1,
        Tensor.__lshift__: lambda self, other: -1,
        Tensor.__ilshift__: lambda self, other: -1,
        Tensor.__rshift__: lambda self, other: -1,
        Tensor.__irshift__: lambda self, other: -1,
        Tensor.__float__: lambda self: -1,
        Tensor.__bool__: lambda self: -1,
        Tensor.__contains__: lambda self, other: -1,
        Tensor.__neg__: lambda self: -1,
        Tensor.__invert__: lambda self: -1,
        Tensor.__mod__: lambda self, other: -1,
        Tensor.__getitem__: lambda self, idx: -1,
        Tensor.__int__: lambda self: -1,
        Tensor.__long__: lambda self: -1,
        Tensor.__hash__: lambda self: -1,
        Tensor.__index__: lambda self: -1,
        Tensor.__format__: lambda self, format_spec: -1,
        Tensor.__reduce_ex__: lambda self, proto: -1,
        Tensor.__repr__: lambda self: -1,
        Tensor.__setitem__: lambda self, k, v: -1,
        Tensor.grad.__get__: lambda self: -1,
        Tensor.is_cuda.__get__: lambda self: -1,
        Tensor.is_leaf.__get__: lambda self: -1,
        Tensor.requires_grad.__get__: lambda self: -1,
        Tensor.type: lambda self, dtype=None, non_blocking=False, **kwargs: -1,
        Tensor._dimI: lambda self: -1,
        Tensor._dimV: lambda self: -1,
        Tensor._indices: lambda self: -1,
        Tensor._nnz: lambda self: -1,
        Tensor._values: lambda self: -1,
        Tensor.apply_: lambda self, callable: -1,
        Tensor.as_strided: lambda self, size, stride: -1,
        Tensor.as_strided_: lambda self, size, stride: -1,
        Tensor.backward: lambda self, gradient=None, retain_graph=None, create_graph=False: -1,
        Tensor.cauchy_: lambda self, median=0, sigma=1, *, generator=None: -1,
        Tensor.coalesce: lambda self: -1,
        Tensor._coalesced_: lambda self, coalesced: -1,
        Tensor.dense_dim: lambda self: -1,
        Tensor.dim: lambda self: -1,
        Tensor.element_size: lambda self: -1,
        Tensor.expand: lambda self, size: -1,
        Tensor.expand_as: lambda self, other: -1,
        Tensor.exponential_: lambda self, lambd=1, *, generator=None: -1,
        Tensor.fill_: lambda self, value: -1,
        Tensor.geometric_: lambda self, p, *, generator=None: -1,
        Tensor.get_device: lambda self: -1,
        Tensor.indices: lambda self: -1,
        Tensor.is_coalesced: lambda self: -1,
        Tensor.is_contiguous: lambda self: -1,
        Tensor.is_pinned: lambda self: -1,
        Tensor.is_set_to: lambda self, tensor: -1,
        Tensor.is_shared: lambda self: -1,
        Tensor.item: lambda self: -1,
        Tensor.log_normal_: lambda self, mean=1, std=2, *, generator=None: -1,
        Tensor.log_softmax: lambda self, dim: -1,
        Tensor.map_: lambda self, tensor, callable: -1,
        Tensor.mm: lambda self, mat2: -1,
        Tensor.narrow_copy: lambda self, dimension, start, length: -1,
        Tensor.ndimension: lambda self: -1,
        Tensor.nelement: lambda self: -1,
        Tensor.normal_: lambda self: -1,
        Tensor.numpy: lambda self: -1,
        Tensor.permute: lambda self, dim: -1,
        Tensor.pin_memory: lambda self: -1,
        Tensor.put_: lambda self, indices, tensor, accumulate=False: -1,
        Tensor.random_: lambda self, from_=0, to=None, *, generator=None: -1,
        Tensor.register_hook: lambda self, hook: -1,
        Tensor.repeat: lambda self, *size: -1,
        Tensor.requires_grad_: lambda self, requires_grad=True: -1,
        Tensor.reshape_as: lambda self, other: -1,
        Tensor.resize_: lambda self, size: -1,
        Tensor.retain_grad: lambda self: -1,
        Tensor.set_: lambda self, source=None, storage_offset=0, size=None, stride=None: -1,
        Tensor.share_memory_: lambda self: -1,
        Tensor.size: lambda self: -1,
        Tensor.sparse_dim: lambda self: -1,
        Tensor.sparse_mask: lambda self, mask: -1,
        Tensor.sparse_resize_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sparse_resize_and_clear_: lambda self, size1, size2, dense_dim: -1,
        Tensor.sspaddmm: lambda self, mat1, mat2, beta=1, alpha=1, out=None: -1,
        Tensor.storage: lambda self: -1,
        Tensor.storage_offset: lambda self: -1,
        Tensor.sum_to_size: lambda self, size: -1,
        Tensor.to_dense: lambda self: -1,
        Tensor.to_sparse: lambda self: -1,
        Tensor.tolist: lambda self: -1,
        Tensor.to_mkldnn: lambda self: -1,
        Tensor.type_as: lambda self, other: -1,
        Tensor.unfold: lambda self, dimension, size, step: -1,
        Tensor.uniform_: lambda self, from_=0, to=1: -1,
        Tensor.values: lambda self: -1,
        Tensor.view: lambda self, shape: -1,
        Tensor.view_as: lambda self, other: -1,
        Tensor.zero_: lambda self: -1,
    }

    if IS_ABOVE_110:
        memory_format = {
            Tensor.byte: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.char: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.to: lambda self, dtype, non_blocking=False, copy=False,
                              memory_format=torch.preserve_format: -1,
            Tensor.short: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.long: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.int: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.half: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.float: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.double: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.contiguous: lambda self, memory_format=torch.contiguous_format: -1,
            Tensor.cpu: lambda self, memory_format=torch.preserve_format: -1,
            Tensor.cuda: lambda self, memory_format=torch.preserve_format: -1,
        }
    else:
        memory_format = {
            Tensor.byte: lambda self: -1,
            Tensor.char: lambda self: -1,
            Tensor.to: lambda self, dtype, non_blocking=False, copy=False: -1,
            Tensor.short: lambda self: -1,
            Tensor.long: lambda self: -1,
            Tensor.int: lambda self: -1,
            Tensor.half: lambda self: -1,
            Tensor.float: lambda self: -1,
            Tensor.double: lambda self: -1,
            Tensor.contiguous: lambda self: -1,
            Tensor.cpu: lambda self: -1,
            Tensor.cuda: lambda self: -1,
        }
    ret.update(memory_format)
    return ret


def get_torch_overrides() -> Dict[Callable, Callable]:

    ret = {
        torch.abs: lambda input, out=None: -1,
        torch.adaptive_avg_pool1d: lambda input, output_size: -1,
        torch.adaptive_max_pool1d: lambda inputs, output_size: -1,
        torch.acos: lambda input, out=None: -1,
        torch.add: lambda input, other, out=None: -1,
        torch.addbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.addcdiv: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addcmul: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.addmv: lambda input, mat, vec, beta=1, alpha=1, out=None: -1,
        torch.addr: lambda input, vec1, vec2, beta=1, alpha=1, out=None: -1,
        torch.affine_grid_generator: lambda theta, size, align_corners: -1,
        torch.all: lambda input, dim=None: -1,
        torch.allclose: lambda input, other, trol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.alpha_dropout: lambda input, p, train, inplace=False: -1,
        torch.any: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.argmax: lambda input: -1,
        torch.argmin: lambda input: -1,
        torch.argsort: lambda input, dim=None: -1,
        torch.asin: lambda input, out=None: -1,
        torch.atan: lambda input, out=None: -1,
        torch.atan2: lambda input, other, out=None: -1,
        torch.avg_pool1d: lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                                 count_include_pad=True: -1,
        torch.baddbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum,
                                 eps, cudnn_enabled: -1,
        torch.batch_norm_backward_elemt: lambda grad_out, input, mean, invstd, weight, mean_dy,
                                                mean_dy_xmu: -1,
        torch.batch_norm_backward_reduce: lambda grad_out, input, mean, invstd, weight, input_g,
                                                 weight_g, bias_g: -1,
        torch.batch_norm_elemt: lambda input, weight, bias, mean, invstd, eps: -1,
        torch.batch_norm_gather_stats: lambda input, mean, invstd, running_mean, running_var,
                                              momentum, eps, count: -1,
        torch.batch_norm_stats: lambda input, eps: -1,
        torch.batch_norm_update_stats: lambda input, running_mean, running_var, momentum: -1,
        torch.bernoulli: lambda input, generator=None, out=None: -1,
        torch.bilinear: lambda input1, input2, weight, bias: -1,
        torch.bincount: lambda input, weights=None, minlength=0: -1,
        torch.bmm: lambda input, mat2, out=None: -1,
        torch.broadcast_tensors: lambda *tensors: -1,
        torch.cartesian_prod: lambda *tensors: -1,
        torch.cat: lambda tensors, dim=0, out=None: -1,
        torch.cdist: lambda x1, c2, p=2, compute_mode=None: -1,
        torch.ceil: lambda input, out=None: -1,
        torch.celu: lambda input, alhpa=1., inplace=False: -1,
        torch.chain_matmul: lambda *matrices: -1,
        torch.cholesky: lambda input, upper=False, out=None: -1,
        torch.cholesky_inverse: lambda input, upper=False, out=None: -1,
        torch.cholesky_solve: lambda input1, input2, upper=False, out=None: -1,
        torch.chunk: lambda input, chunks, dim=0: -1,
        torch.clamp: lambda input, min=None, max=None, out=None: -1,
        torch.clamp_min: lambda input, min, out=None: -1,
        torch.clamp_max: lambda input, max, out=None: -1,
        torch.clone: lambda input: -1,
        torch.combinations: lambda input, r=2, with_replacement=False: -1,
        torch.constant_pad_nd: lambda input, pad, value=0: -1,
        torch.conv1d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1,
                             groups=1: -1,
        torch.conv2d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1,
                             groups=1: -1,
        torch.conv3d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1,
                             groups=1: -1,
        torch.convolution: lambda input, weight, bias, stride, padding, dilation, transposed,
                                  output_adding, groups: -1,
        torch.conv_tbc: lambda input, weight, bias, pad=0: -1,
        torch.conv_transpose1d: lambda input, weight, bias=None, stride=1, padding=0,
                                       output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose2d: lambda input, weight, bias=None, stride=1, padding=0,
                                       output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose3d: lambda input, weight, bias=None, stride=1, padding=0,
                                       output_padding=0, groups=1, dilation=1: -1,
        torch.cos: lambda input, out=None: -1,
        torch.cosh: lambda input, out=None: -1,
        torch.cosine_similarity: lambda x1, x2, dim=1, eps=1e-8: -1,
        torch.cross: lambda input, other, dim=-1, out=None: -1,
        torch.cumprod: lambda input, dim, out=None, dtype=None: -1,
        torch.cumsum: lambda input, dim, out=None, dtype=None: -1,
        torch.det: lambda input: -1,
        torch.detach: lambda input: -1,
        torch.diag: lambda input, diagonal=0, out=None: -1,
        torch.diag_embed: lambda input, diagonal=0, out=None: -1,
        torch.diagflat: lambda input, offset=0: -1,
        torch.diagonal: lambda input, offset=0, dim1=0, dim2=1: -1,
        torch.digamma: lambda input, out=None: -1,
        torch.dist: lambda input, other, p=2: -1,
        torch.div: lambda input, other, out=None: -1,
        torch.dot: lambda mat1, mat2: -1,
        torch.dropout: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.eig: lambda input, eigenvectors=False, out=None: -1,
        torch.einsum: lambda equation, *operands: -1,
        torch.embedding: (lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                                 scale_grad_by_freq=False,
                                 sparse=False: -1),
        torch.embedding_bag: (
            lambda input, weight, offsets, max_norm=None, norm_type=2, scale_grad_by_freq=False,
                   mode='mean', sparse=False, per_sample_weights=None: -1),
        torch.empty_like: lambda input, dtype=None, layout=None, device=None,
                                 requires_grad=False: -1,
        torch.eq: lambda input, other, out=None: -1,
        torch.equal: lambda input, other: -1,
        torch.erf: lambda input, out=None: -1,
        torch.erfc: lambda input, out=None: -1,
        torch.erfinv: lambda input, out=None: -1,
        torch.exp: lambda input, out=None: -1,
        torch.expm1: lambda input, out=None: -1,
        torch.fbgemm_linear_int8_weight: lambda input, weight, packed, col_offsets, weight_scale,
                                                weight_zero_point, bias: -1,
        torch.feature_alpha_dropout: lambda input, p, train: -1,
        torch.feature_dropout: lambda input, p, train: -1,
        torch.flatten: lambda input, start_dim=0, end_dim=-1: -1,
        torch.flip: lambda input, dims: -1,
        torch.frobenius_norm: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.floor: lambda input, out=None: -1,
        torch.fmod: lambda input, other, out=None: -1,
        torch.frac: lambda input, out=None: -1,
        torch.full_like: lambda input, fill_value, out=None, dtype=None, layout=torch.strided,
                                device=None, requires_grad=False: -1,
        torch.functional.lu_unpack: lambda LU_data, LU_pivots, unpack_data=True,
                                           unpack_pivots=True: -1,
        torch.gather: lambda input, dim, index, out=None, sparse_grad=False: -1,
        torch.ge: lambda input, other, out=None: -1,
        torch.geqrf: lambda input, out=None: -1,
        torch.ger: lambda input, vec2, out=None: -1,
        torch.grid_sampler: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.grid_sampler_2d: lambda input, grid, interpolation_mode, padding_mode,
                                      align_corners: -1,
        torch.grid_sampler_3d: lambda input, grid, interpolation_mode, padding_mode,
                                      align_corners: -1,
        torch.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05,
                                 cudnn_enabled=True: -1,
        torch.gru: lambda input, hx, params, has_biases, num_layers, gropout, train, bidirectional,
                          batch_first: -1,
        torch.gru_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.gt: lambda input, other, out=None: -1,
        torch.hardshrink: lambda input, lambd=0.5: -1,
        torch.histc: lambda input, bins=100, min=0, max=0, out=None: -1,
        torch.hspmm: lambda mat1, mat2, out=None: -1,
        torch.index_add: lambda input, dim, index, source: -1,
        torch.index_copy: lambda input, dim, index, source: -1,
        torch.index_put: lambda input, indices, values, accumulate=False: -1,
        torch.index_select: lambda input, dim, index, out=None: -1,
        torch.index_fill: lambda input, dim, index, value: -1,
        torch.isfinite: lambda tensor: -1,
        torch.isinf: lambda tensor: -1,
        torch.instance_norm: (
            lambda input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps,
                   cudnn_enabled: -1),
        torch.int_repr: lambda input: -1,
        torch.inverse: lambda input, out=None: -1,
        torch.is_complex: lambda input: -1,
        torch.is_distributed: lambda input: -1,
        torch.is_floating_point: lambda input: -1,
        torch.is_nonzero: lambda input: -1,
        torch.is_same_size: lambda input, other: -1,
        torch.is_signed: lambda input: -1,
        torch.isclose: lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.isnan: lambda input: -1,
        torch.kthvalue: lambda input, k, dim=None, keepdim=False, out=None: -1,
        torch.layer_norm: lambda input, normalized_shape, weight=None, bias=None, esp=1e-05,
                                 cudnn_enabled=True: -1,
        torch.le: lambda input, other, out=None: -1,
        torch.lerp: lambda input, end, weight, out=None: -1,
        torch.lgamma: lambda input, out=None: -1,
        torch.log: lambda input, out=None: -1,
        torch.log_softmax: lambda input, dim, dtype=None: -1,
        torch.log10: lambda input, out=None: -1,
        torch.log1p: lambda input, out=None: -1,
        torch.log2: lambda input, out=None: -1,
        torch.logdet: lambda input: -1,
        torch.lstm: lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train,
                           bidirectional: -1,
        torch.lstm_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.lt: lambda input, other, out=None: -1,
        torch.lu: lambda A, pivot=True, get_infos=False, out=None: -1,
        torch.lu_solve: lambda input, LU_data, LU_pivots, out=None: -1,
        torch.masked_fill: lambda input, mask, value: -1,
        torch.masked_scatter: lambda input, mask, source: -1,
        torch.masked_select: lambda input, mask, out=None: -1,
        torch.matmul: lambda input, other, out=None: -1,
        torch.matrix_power: lambda input, n: -1,
        torch.matrix_rank: lambda input, tol=None, symmetric=False: -1,
        torch.max: lambda input, dim=None, out=None: -1,
        torch.max_pool1d: lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                 ceil_mode=False: -1,
        torch.max_pool2d: lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                 ceil_mode=False: -1,
        torch.max_pool3d: lambda input, kernel_size, stride=None, padding=0, dilation=1,
                                 ceil_mode=False: -1,
        torch.max_pool1d_with_indices: lambda input, kernel_size, stride=None, padding=0,
                                              dilation=1, ceil_mode=False: -1,
        torch.mean: lambda input, dim=None: -1,
        torch.median: lambda input, dim=None: -1,
        torch.meshgrid: lambda *tensors, **kwargs: -1,
        torch.min: lambda input, dim=None, out=None: -1,
        torch.miopen_batch_norm: (lambda input, weight, bias, running_mean, running_var, training,
                                         exponential_average_factor, epsilon: -1),
        torch.miopen_convolution: lambda input, weight, bias, padding, stride, dilation, groups,
                                         benchmark, deterministic: -1,
        torch.miopen_convolution_transpose: (
            lambda input, weight, bias, padding, output_padding, stride, dilation,
                   groups, benchmark, deterministic: -1),
        torch.miopen_depthwise_convolution: (
            lambda input, weight, bias, padding, stride, dilation, groups, benchmark,
                   deterministic: -1),
        torch.mm: lambda input, mat2, out=None: -1,
        torch.mode: lambda input, dim=-1, keepdim=False, out=None: -1,
        torch.mul: lambda input, other, out=None: -1,
        torch.multinomial: lambda input, num_samples, replacement=False, out=None: -1,
        torch.mv: lambda input, vec, out=None: -1,
        torch.mvlgamma: lambda input, p: -1,
        torch.narrow: lambda input, dim, start, length: -1,
        torch.native_batch_norm: lambda input, weight, bias, running_mean, running_var, training,
                                        momentum, eps: -1,
        torch.ne: lambda input, other, out=None: -1,
        torch.neg: lambda input, out=None: -1,
        torch.nonzero: lambda input, as_tuple=False: -1,
        torch.norm: lambda input, p='fro', dim=None, keepdim=False, out=None, dtype=None: -1,
        torch.norm_except_dim: lambda v, pow=2, dim=0: -1,
        torch.nuclear_norm: lambda input, p='fro', dim=None, keepdim=False, out=None,
                                   dtype=None: -1,
        torch.numel: lambda input: -1,
        torch.orgqr: lambda input1, input2: -1,
        torch.ormqr: lambda input, input2, input3, left=True, transpose=False: -1,
        torch.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        torch.pdist: lambda input, p=2: -1,
        torch.pinverse: lambda input, rcond=1e-15: -1,
        torch.pixel_shuffle: lambda input, upscale_factor: -1,
        torch.poisson: lambda input, generator=None: -1,
        torch.polygamma: lambda input, n, out=None: -1,
        torch.prelu: lambda input, weight: -1,
        torch.ones_like: lambda input, dtype=None, layout=None, device=None,
                                requires_grad=False: -1,
        torch.pow: lambda input, exponent, out=None: -1,
        torch.prod: lambda input, dtype=None: -1,
        torch.q_scale: lambda input: -1,
        torch.q_zero_point: lambda input: -1,
        torch.qr: lambda input, some=True, out=None: -1,
        torch.rand_like: lambda input, dtype=None, layout=None, device=None,
                                requires_grad=False: -1,
        torch.randint_like: lambda input, high, dtype=None, layout=torch.strided, device=None,
                                   requires_grad=False: -1,
        torch.randn_like: lambda input, dtype=None, layout=None, device=None,
                                 requires_grad=False: -1,
        torch.reciprocal: lambda input, out=None: -1,
        torch.relu: lambda input, inplace=False: -1,
        torch.remainder: lambda input, other, out=None: -1,
        torch.renorm: lambda input, p, dim, maxnorm, out=None: -1,
        torch.repeat_interleave: lambda input, dim=None: -1,
        torch.reshape: lambda input, shape: -1,
        torch.rnn_relu: lambda input, hx, params, has_biases, num_layers, dropout, train,
                               bidirectional, batch_first: -1,
        torch.rnn_relu_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.rnn_tanh: lambda input, hx, params, has_biases, num_layers, dropout, train,
                               bidirectional, batch_first: -1,
        torch.rnn_tanh_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.roll: lambda input, shifts, dims=None: -1,
        torch.rot90: lambda input, k=1, dims=(0, 1): -1,
        torch.round: lambda input, out=None: -1,
        torch.rrelu: lambda input, lower=1. / 8, upper=1. / 3, training=False, inplace=False: -1,
        torch.rsqrt: lambda input, out=None: -1,
        torch.rsub: lambda input, other, alpha=1: -1,
        torch.scatter: lambda input, dim, index, src: -1,
        torch.scatter_add: lambda input, dim, index, src: -1,
        torch.select: lambda input, dim, index: -1,
        torch.selu: lambda input, inplace=False: -1,
        torch.sigmoid: lambda input, out=None: -1,
        torch.sign: lambda input, out=None: -1,
        torch.sin: lambda input, out=None: -1,
        torch.sinh: lambda input, out=None: -1,
        torch.slogdet: lambda input: -1,
        torch.smm: lambda input, mat2: -1,
        torch.softmax: lambda input, dim, dtype=None: -1,
        torch.solve: lambda input, A, out=None: -1,
        torch.sort: lambda input, dim=-1, descending=False, out=None: -1,
        torch.split: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.split_with_sizes: lambda tensor, split_size_or_sections, dim=0: -1,
        torch.sqrt: lambda input, out=None: -1,
        torch.squeeze: lambda input, dim=None, out=None: -1,
        torch.sspaddmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.stack: lambda tensors, dim=0, out=None: -1,
        torch.std: lambda input, dim=None: -1,
        torch.sub: lambda input, other, out=None: -1,
        torch.sum: lambda input, dim=None: -1,
        torch.svd: lambda input, some=True, compute_uv=True, out=None: -1,
        torch.symeig: lambda input, eigenvectors=False, upper=True, out=None: -1,
        torch.t: lambda input: -1,
        torch.take: lambda input, index: -1,
        torch.tan: lambda input, out=None: -1,
        torch.tanh: lambda input, out=None: -1,
        torch.tensordot: lambda a, b, dims=2: -1,
        torch.threshold: lambda input, threshold, value, inplace=False: -1,
        torch.topk: lambda input, k, dim=-1, descending=False, out=None: -1,
        torch.trace: lambda input: -1,
        torch.transpose: lambda input, dim0, dim1: -1,
        torch.triangular_solve: lambda input, A, upper=True, transpose=False,
                                       unitriangular=False: -1,
        torch.tril: lambda input, diagonal=0, out=None: -1,
        torch.triu: lambda input, diagonal=0, out=None: -1,
        torch.trunc: lambda input, out=None: -1,
        torch.unbind: lambda input, dim=0: -1,
        torch.unique: lambda input, sorted=True, return_inverse=False, return_counts=False,
                             dim=None: -1,
        torch.unsqueeze: lambda input, dim, out=None: -1,
        torch.var: lambda input, dim=None: -1,
        torch.where: lambda condition, x=None, y=None: -1,
        torch.zeros_like: lambda input, dtype=None, layout=None, device=None,
                                 requires_grad=False: -1,
    }

    try:
        ffts = {
            torch.fft: lambda input, signal_ndim, normalized=False: -1,
            torch.rfft: lambda input, signal_ndim, normalized=False, onesided=True: -1,
            torch.ifft: lambda input, signal_ndim, normalized=False: -1,
            torch.irfft: lambda input, signal_ndim, normalized=False, onesided=True,
                            signal_sizes=None: -1,
            torch.stft: (
                lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True,
                   pad_mode='reflect', normalized=False, onesided=True, return_complex=None: -1),
        }
    # unsupported above 1.8.0
    except AttributeError:
        ffts = {}

    ret.update(ffts)

    return ret


def get_nn_functional_overrides() -> Dict[Callable, Callable]:

    ret = {
        torch.nn.functional.adaptive_avg_pool2d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_avg_pool3d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_max_pool1d: lambda input, output_size,
                                                        return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool1d_with_indices: lambda input, output_size,
                                                                     return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d: lambda input, output_size,
                                                        return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d_with_indices: lambda input, output_size,
                                                                     return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d: lambda input, output_size,
                                                        return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d_with_indices: lambda input, output_size,
                                                                     return_indices=False: -1,
        torch.nn.functional.affine_grid: lambda theta, size, align_corners=None: -1,
        torch.nn.functional.alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        torch.nn.functional.batch_norm: (
            lambda input, running_mean, running_var, weight=None, bias=None, training=False,
                   momentum=0.1, eps=1e-05: -1),
        torch.nn.functional.bilinear: lambda input1, input2, weight, bias=None: -1,
        torch.nn.functional.binary_cross_entropy: (
            lambda input, target, weight=None, size_average=None, reduce=None,
                   reduction="mean": -1),
        torch.nn.functional.binary_cross_entropy_with_logits: (
            lambda input, target, weight=None, size_average=None,
                   reduce=None, reduction="mean", pos_weight=None: -1),
        torch.nn.functional.celu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.conv1d: lambda input, weight, bias=None, stride=1, padding=0,
                                           dilation=1,
                                           groups=1: -1,
        torch.nn.functional.conv2d: lambda input, weight, bias=None, stride=1, padding=0,
                                           dilation=1,
                                           groups=1: -1,
        torch.nn.functional.conv3d: lambda input, weight, bias=None, stride=1, padding=0,
                                           dilation=1,
                                           groups=1: -1,
        torch.nn.functional.conv_tbc: lambda input, weight, bias, pad=0: -1,
        torch.nn.functional.conv_transpose1d: lambda input, weight, bias=None, stride=1, padding=0,
                                                     output_padding=0, groups=1, dilation=1: -1,
        torch.nn.functional.conv_transpose2d: lambda input, weight, bias=None, stride=1, padding=0,
                                                     output_padding=0, groups=1, dilation=1: -1,
        torch.nn.functional.conv_transpose3d: lambda input, weight, bias=None, stride=1, padding=0,
                                                     output_padding=0, groups=1, dilation=1: -1,
        torch.nn.functional.cosine_embedding_loss: (
            lambda input1, input2, target, margin=0, size_average=None,
                   reduce=None, reduction='mean': -1),
        torch.nn.functional.cross_entropy: (
            lambda input, target, weight=None, size_average=None, ignore_index=-100,
                   reduce=None, reduction="mean": -1),
        torch.nn.functional.ctc_loss: (
            lambda log_probs, targets, input_lengths, target_lengths, blank=0,
                   reduction='mean', zero_infinity=False: -1),
        torch.nn.functional.dropout: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout2d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout3d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.elu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.embedding: (
            lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                   scale_grad_by_freq=False, sparse=False: -1),
        torch.nn.functional.embedding_bag: (
            lambda input, weight, offsets=None, max_norm=None, norm_type=2,
                   scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None,
                   include_last_offset=False: -1),
        torch.nn.functional.feature_alpha_dropout: lambda input, p=0.5, training=False,
                                                          inplace=False: -1,
        torch.nn.functional.fold: lambda input, output_size, kernel_size, dilation=1, padding=0,
                                         stride=1: -1,
        torch.nn.functional.fractional_max_pool2d: (
            lambda input, kernel_size, output_size=None, output_ratio=None,
                   return_indices=False, _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool2d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
                   _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool3d: (
            lambda input, kernel_size, output_size=None, output_ratio=None,
                   return_indices=False, _random_samples=None: -1),
        torch.nn.functional.fractional_max_pool3d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
                   _random_samples=None: -1),
        torch.nn.functional.glu: lambda input, dim=-1: -1,
        torch.nn.functional.grid_sample: lambda input, grid, mode='bilinear', padding_mode='zeros',
                                                align_corners=None: -1,
        torch.nn.functional.group_norm: lambda input, num_groups, weight=None, bias=None,
                                               eps=1e-05: -1,
        torch.nn.functional.gumbel_softmax: lambda logits, tau=1, hard=False, eps=1e-10, dim=-1: -1,
        torch.nn.functional.hardshrink: lambda input, lambd=0.5: -1,
        torch.nn.functional.hardtanh: lambda input, min_val=-1., max_val=1., inplace=False: -1,
        torch.nn.functional.hinge_embedding_loss: (
            lambda input, target, margin=1.0, size_average=None, reduce=None,
                   reduction='mean': -1),
        torch.nn.functional.instance_norm: (
            lambda input, running_mean=None, running_var=None, weight=None, bias=None,
                   use_input_stats=True, momentum=0.1, eps=1e-05: -1),
        torch.nn.functional.interpolate: (
            lambda input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                   recompute_scale_factor=None: -1),
        torch.nn.functional.kl_div: lambda input, target, size_average=None, reduce=None,
                                           reduction='mean', log_target=False: -1,
        torch.nn.functional.l1_loss: lambda input, target, size_average=None, reduce=None,
                                            reduction='mean': -1,
        torch.nn.functional.layer_norm: lambda input, normalized_shape, weight=None, bias=None,
                                               eps=1e-05: -1,
        torch.nn.functional.leaky_relu: lambda input, negative_slope=0.01, inplace=False: -1,
        torch.nn.functional.linear: lambda input, weight, bias=None: -1,
        torch.nn.functional.local_response_norm: lambda input, size, alpha=0.0001, beta=0.75,
                                                        k=1.0: -1,
        torch.nn.functional.log_softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.lp_pool1d: lambda input, norm_type, kernel_size, stride=None,
                                              ceil_mode=False: -1,
        torch.nn.functional.lp_pool2d: lambda input, norm_type, kernel_size, stride=None,
                                              ceil_mode=False: -1,
        torch.nn.functional.margin_ranking_loss: (
            lambda input1, input2, target, margin=0, size_average=None,
                   reduce=None, reduction='mean': -1),
        torch.nn.functional.max_pool1d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool1d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool2d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool2d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool3d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_pool3d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False: -1),
        torch.nn.functional.max_unpool1d: lambda input, indices, kernel_size, stride=None,
                                                 padding=0, output_size=None: -1,
        torch.nn.functional.max_unpool2d: lambda input, indices, kernel_size, stride=None,
                                                 padding=0, output_size=None: -1,
        torch.nn.functional.max_unpool3d: lambda input, indices, kernel_size, stride=None,
                                                 padding=0, output_size=None: -1,
        torch.nn.functional.mse_loss: lambda input, target, size_average=None, reduce=None,
                                             reduction='mean': -1,
        torch.nn.functional.multi_margin_loss: (
            lambda input, target, p=1, margin=1.0, weight=None, size_average=None,
                   reduce=None, reduction='mean': -1),
        torch.nn.functional.multilabel_margin_loss: (
            lambda input, target, size_average=None, reduce=None,
                   reduction='mean': -1),
        torch.nn.functional.multilabel_soft_margin_loss: (
            lambda input, target, weight=None, size_average=None,
                   reduce=None, reduction='mean': -1),
        torch.nn.functional.nll_loss: (
            lambda input, target, weight=None, size_average=None, ignore_index=-100,
                   reduce=None, reduction='mean': -1),
        torch.nn.functional.normalize: lambda input, p=2, dim=1, eps=1e-12, out=None: -1,
        torch.nn.functional.one_hot: lambda tensor, num_classes=-1: -1,
        torch.nn.functional.pad: lambda input, pad, mode='constant', value=0: -1,
        torch.nn.functional.pairwise_distance: lambda x1, x2, p=2.0, eps=1e-06, keepdim=False: -1,
        torch.nn.functional.poisson_nll_loss: (
            lambda input, target, log_input=True, full=False, size_average=None,
                   eps=1e-08, reduce=None, reduction='mean': -1),
        torch.nn.functional.prelu: lambda input, weight: -1,
        torch.nn.functional.relu: lambda input, inplace=False: -1,
        torch.nn.functional.relu6: lambda input, inplace=False: -1,
        torch.nn.functional.rrelu: lambda input, lower=0.125, upper=0.3333333333333333,
                                          training=False, inplace=False: -1,
        torch.nn.functional.selu: lambda input, inplace=False: -1,
        torch.nn.functional.smooth_l1_loss: lambda input, target, size_average=None, reduce=None,
                                                   reduction='mean', beta=1.: -1,
        torch.nn.functional.soft_margin_loss: lambda input, target, size_average=None, reduce=None,
                                                     reduction='mean': -1,
        torch.nn.functional.softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.softmin: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.softplus: lambda input, beta=1, threshold=20: -1,
        torch.nn.functional.softshrink: lambda input, lambd=0.5: -1,
        torch.nn.functional.softsign: lambda input: -1,
        torch.nn.functional.tanhshrink: lambda input: -1,
        torch.nn.functional.threshold: lambda input, threshold, value, inplace=False: -1,
        torch.nn.functional.triplet_margin_loss: (
            lambda anchor, positive, negative, margin=1.0, p=2, eps=1e-06,
                   swap=False, size_average=None, reduce=None, reduction='mean': -1),
        torch.nn.functional.unfold: lambda input, kernel_size, dilation=1, padding=0, stride=1: -1,

    }

    if IS_ABOVE_110:
        func_avg_pool = {
            torch.nn.functional.avg_pool1d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True, divisor_override=None: -1),
            torch.nn.functional.avg_pool2d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True, divisor_override=None: -1),
            torch.nn.functional.avg_pool3d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True, divisor_override=None: -1)
        }

    else:
        func_avg_pool = {
            torch.nn.functional.avg_pool1d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True: -1),
            torch.nn.functional.avg_pool2d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True: -1),
            torch.nn.functional.avg_pool3d: (
                lambda input, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True: -1)
        }

    ret.update(func_avg_pool)
    return ret


@functools.lru_cache(None)
def get_testing_overrides() -> Dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    A dictionary that maps overridable functions in the PyTorch API to
    lambda functions that have the same signature as the real function
    and unconditionally return -1. These lambda functions are useful
    for testing API coverage for a type that defines __torch_function__.
    """
    # Every function in the PyTorch API that can be overriden needs an entry
    # in this dict.
    #
    # Optimally we would use inspect to get the function signature and define
    # the lambda function procedurally but that is blocked by generating
    # function signatures for native kernels that can be consumed by inspect.
    # See Issue #28233.
    ret = {}
    ret.update(get_tensor_overrides())
    ret.update(get_torch_overrides())
    ret.update(get_nn_functional_overrides())
    return ret