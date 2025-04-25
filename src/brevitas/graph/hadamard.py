# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math
import os
import pathlib

try:
    import fast_hadamard_transform
except:
    fast_hadamard_transform = None
import torch

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py


def get_hadK(n, transpose=False):
    parent = pathlib.Path(os.path.abspath(__file__)).parent
    # hadamard matrices for had12, had36.pal2, had52,will,
    # # had60.pal, had108.pal, had140.pal, had156.will, had172.will:
    # http://www.neilsloane.com/hadamard/index.html
    tensors = torch.load(str(parent) + '/hadamard_tensors.pt', weights_only=True)
    tensors = {k: v.to(torch.float) for k, v in tensors.items()}
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert (is_pow2(n // 172))
        K = 172
        hadK = tensors['get_had172'].T if transpose else tensors['get_had172']
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert (is_pow2(n // 156))
        K = 156
        hadK = tensors['get_had156'].T if transpose else tensors['get_had156']
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert (is_pow2(n // 140))
        K = 140
        hadK = tensors['get_had140'].T if transpose else tensors['get_had140']
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert (is_pow2(n // 108))
        K = 108
        hadK = tensors['get_had108'].T if transpose else tensors['get_had108']
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert (is_pow2(n // 60))
        K = 60
        hadK = tensors['get_had60'].T if transpose else tensors['get_had60']
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert (is_pow2(n // 52))
        K = 52
        hadK = tensors['get_had52'].T if transpose else tensors['get_had52']
    elif n % 36 == 0:
        assert (is_pow2(n // 36))
        K = 36
        hadK = tensors['get_had36'].T if transpose else tensors['get_had36']
    elif n % 28 == 0:
        assert (is_pow2(n // 28))
        K = 28
        hadK = tensors['get_had28'].T if transpose else tensors['get_had28']
    elif n % 40 == 0:
        assert (is_pow2(n // 40))
        K = 40
        hadK = tensors['get_had40'].T if transpose else tensors['get_had40']
    elif n % 20 == 0:
        assert (is_pow2(n // 20))
        K = 20
        hadK = tensors['get_had20'].T if transpose else tensors['get_had20']
    elif n % 12 == 0:
        assert (is_pow2(n // 12))
        K = 12
        hadK = tensors['get_had12'].T if transpose else tensors['get_had12']
    else:
        assert (is_pow2(n))
        K = 1

    return hadK, K


def find_closest_hadamard_number(starting_dim, steps=1):
    import math

    values_to_check = [172, 156, 140, 108, 60, 52, 40, 36, 28, 20, 12]

    for step in range(steps):
        best_value = None
        next_dim = starting_dim + 1
        for v in values_to_check:
            m = math.ceil(next_dim / v) * v
            m = torch.tensor(math.ceil(next_dim / v))
            floor_po2_m = torch.pow(2, torch.log2(m).floor())
            ceil_po2_m = torch.pow(2, torch.log2(m).ceil())
            m = floor_po2_m * v if floor_po2_m > starting_dim else ceil_po2_m * v
            if best_value is None:
                best_value = m
            else:
                best_value = m if (
                    abs(starting_dim - m) <= abs(starting_dim - best_value) and
                    m > starting_dim) else best_value
        starting_dim = best_value

    return int(best_value.cpu().item())


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().reshape(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def random_hadamard_matrix(size, device):
    # See https://github.com/Cornell-RelaxML/quip-sharp , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    # Set to float32 for consistency with random_orthogonal_matrix and get_hadK
    return matmul_hadU(Q).to(device).float()


def matmul_hadU_cuda(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(
            X.contiguous(), 1.0 / torch.tensor(n).sqrt())
    # if transpose:
    #     hadK = hadK.T.contiguous()
    input = X.view(*X.shape[:-1], K, n // K)
    input = fast_hadamard_transform.hadamard_transform(
        input.contiguous(), 1.0 / torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)


def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        # Apply Hadamard to the last had_dim chunks of the weights
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            W_ = fast_hadamard_transform.hadamard_transform(
                W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim),
                scale=1 / math.sqrt(had_dim)).reshape(transposed_shape).t()
        else:
            raise NotImplementedError("Not implemented (or tested) yet!")
            n = W_.shape[1]
            W_ = hadamard_transform(
                W_.reshape(-1, n // had_dim, had_dim),
                scale=1 / math.sqrt(had_dim)).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
