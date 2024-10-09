# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import List, Optional

import numpy as np
from packaging import version
import torch
from torch import Tensor
import torch.nn as nn

try:
    from torch.linalg import LinAlgError
except:
    LinAlgError = RuntimeError
import warnings

import unfoldNd

from brevitas import torch_version
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import StopFwdException
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.gpxq import SUPPORTED_TCONV_OP
import brevitas.nn as qnn
from brevitas.quant_tensor import _unpack_quant_tensor


class gpfq_mode(gpxq_mode):
    """
    Apply GPFQ algorithm.

    Args:
        model (Module): The model to quantize with GPFQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPFQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gpfq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPFQ. Default: False
        p (float): The percentage of processed inputs to use. Default: 1.0
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with gpfq_mode(model) as gpfq:
        >>>         gpfq_model = gpfq.model
        >>>         for i in tqdm(range(gpfq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gpfq_model(img)
        >>>             gpfq.update()
    """

    def __init__(
            self,
            model: nn.Module,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            p: float = 1.0,
            return_forward_output: bool = False,
            act_order: bool = False,
            gpfq_class: Optional[nn.Module] = None) -> None:
        if not inplace:
            model = deepcopy(model)
        super().__init__(
            model,
            group_of_parallel_layers,
            inplace,
            create_weight_orig,
            use_quant_activations,
            act_order,
            return_forward_output)

        self.p = p
        if gpfq_class is None:
            gpfq_class = GPFQ
        self.gpfq_class = gpfq_class
        assert issubclass(gpfq_class, GPxQ), \
            "Error: expected `gpfq_class` to be derived from `brevitas.graph.gpxq.GPxQ`."

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Disable quantization
        self.return_quant_tensor_state = disable_return_quant_tensor(self.model)
        self.disable_quant_inference.disable_param_quantization(self.model, is_training=False)
        self.disable_quant_inference.disable_act_quantization(self.model, is_training=False)
        # Collect float input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        # Re-enable quantization. If activation quantization is disabled,
        # we also disable bias quantization
        self.disable_quant_inference.enable_param_quantization(self.model, is_training=False)
        if self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(self.model, is_training=False)
        else:
            self.disable_quant_inference.disable_bias_quantization(self.model, is_training=False)
        restore_return_quant_tensor(self.model, self.return_quant_tensor_state)

        if self.return_forward_output:
            # If we want to return the output of the network, we need to disable all hooks
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = True
            out = self.orig_forward(*args, **kwargs)
            for name, gpxq_class in self.gpxq_layers.items():
                gpxq_class.disable_pre_forward_hook = False
            return out

    def initialize_module_optimizer(
            self, layer, name, act_order, len_parallel_layers, create_weight_orig):
        return self.gpfq_class(
            layer=layer,
            name=name,
            act_order=act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            p=self.p)


class GPFQ(GPxQ):
    """
    Based on https://github.com/YixuanSeanZhou/Quantized_Neural_Nets/tree/main
    """

    def __init__(self, layer, name, act_order, len_parallel_layers, create_weight_orig, p) -> None:

        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig)

        self.float_input = None
        self.quant_input = None
        self.index_computed = False
        self.p = p

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        is_quant_enabled = module.weight_quant.is_quant_enabled

        inp = self.process_input(input)
        batch_size = inp.shape[0]

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                unfold_impl = unfoldNd.UnfoldTransposeNd
            else:
                unfold_impl = unfoldNd.UnfoldNd

            unfold = unfold_impl(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.kernel_size)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for i, inp in enumerate(inp_by_group):

                inp = unfold(inp)

                batch_size, num_blocks = inp.shape[0], inp.shape[-1]
                inp = torch.transpose(inp, 1, 2)  # shape (B, L, C*kernel_size[0]*kernel_size[1])
                inp = inp.reshape(-1, inp.size(-1))  # shape (B*L, C*kernel_size[0]*kernel_size[1])

                if not self.index_computed:
                    self.index_computed = True
                    self.rand_indices = np.concatenate([
                        np.random.choice(
                            np.arange(num_blocks * i, num_blocks * (i + 1)),
                            size=int(
                                self.p * num_blocks + 1 if self.p != 1 else self.p * num_blocks))
                        for i in range(batch_size)])  # need to define self.p (probability)

                indexes = self.rand_indices
                if np.max(self.rand_indices) > inp.shape[0]:
                    indexes = self.rand_indices < inp.shape[0]
                    indexes = self.rand_indices[indexes]

                inp = inp[indexes]
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        if not is_quant_enabled:
            if self.float_input is None:
                self.float_input = inp_processed
            else:
                self.float_input = torch.cat([self.float_input, inp_processed], dim=1)
        else:
            if self.quant_input is None:
                self.quant_input = inp_processed
            else:
                self.quant_input = torch.cat([self.quant_input, inp_processed], dim=1)
        # If we are executing GPFQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require quantized inputs."
        weight = self.layer.weight.data
        dev = weight.device
        dtype = weight.dtype
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

        self.float_input = self.float_input.to(dev)
        self.quant_input = self.quant_input.to(dev)
        U = torch.zeros(
            weight.shape[0], weight.shape[1], self.float_input.shape[1], device=dev, dtype=dtype)
        # We don't need full Hessian, we just need the diagonal
        # Summing over batch dimension
        H_diag = self.quant_input.transpose(2, 1).square().sum(2)
        permutation_list = []
        for group_index in range(self.groups):
            if self.act_order:
                # Re-order Hessian_diagonal so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(H_diag[group_index, :], descending=True)
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(weight.shape[-1]), device=dev)
            permutation_list.append(perm)
        del H_diag
        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                U[group_index] += torch.matmul(
                    weight[group_index, :, permutation_list[group_index][t]].unsqueeze(1),
                    self.float_input[group_index, :, permutation_list[group_index][t]].unsqueeze(
                        0))  #[OC/Groups, 1] * [1, INSHAPE[1]]
                norm = torch.linalg.norm(
                    self.quant_input[group_index, :, permutation_list[group_index][t]], 2) ** 2
                if norm > 0:
                    q_arg = U[group_index].matmul(
                        self.quant_input[group_index, :, permutation_list[group_index][t]]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])

                weight[group_index, :, permutation_list[group_index][t]] = q_arg
            q = self.get_quant_weights(t, 0, permutation_list)
            for group_index in range(self.groups):
                U[group_index] -= torch.matmul(
                    q[group_index].unsqueeze(1),
                    self.quant_input[group_index, :, permutation_list[group_index][t]].unsqueeze(0))

        del self.float_input
        del self.quant_input


class GPFQv2(GPFQ):
    """
    Memory-efficient GPFQ formulation introduced in https://arxiv.org/pdf/2409.17092
    """

    def __init__(self, layer, name, act_order, len_parallel_layers, create_weight_orig, p) -> None:
        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig, p)
        # Initialize covariance matrices. We need it in float32 to compute the inverse
        # H = (\hat{X} \hat{X}^T)^{1/2}
        self.H: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32)
        # G = X \hat{X}^T
        self.G: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32)
        # buffer to speed-up GPU to CPU transfer
        self.B: Tensor = torch.zeros((self.groups, self.columns, self.columns),
                                     device="cpu",
                                     dtype=torch.float32,
                                     pin_memory=torch.cuda.is_available())
        self.nsamples = 0

        assert torch_version >= version.parse('1.10'), "GPFQv2 requires torch 1.10 or higher"

    def update_batch(self, module, input, current_layer):
        if self.disable_pre_forward_hook:
            return input

        # Update reference to current layer
        current_layer.layer_names.add(self.name)
        is_quant_enabled = module.weight_quant.is_quant_enabled

        inp = self.process_input(input)
        inp = _unpack_quant_tensor(inp)

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            inp = inp.t()
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                unfold_impl = unfoldNd.UnfoldTransposeNd
            else:
                unfold_impl = unfoldNd.UnfoldNd

            unfold = unfold_impl(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for inp in inp_by_group:
                inp = unfold(inp)
                inp = inp.transpose(1, 0)
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        # NOTE: in the gpfq_mode context manager, we first collect quant inputs, then
        # we collect float inputs for the same batch. We assume this pattern here, but
        # will add a check just in case.
        n = inp_processed.size(1)
        inp_processed = math.sqrt(2 / n) * inp_processed.to(torch.float32)

        # if quant is not enabled, then it is the float input; if it is a float input
        # then a quant input has already happened and we can update G
        if not is_quant_enabled:
            # Computing the normalized H matrix using CPU buffer
            self.B.copy_(self.quant_input.bmm(inp_processed.transpose(2, 1)))
            self.G += self.B
            self.quant_input = None  # NOTE: set back to None now that we've used it
        else:
            # Computing the normalized H matrix using CPU buffer
            self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
            self.H += self.B
            # store the quantized input for computing the H matrix
            assert self.quant_input is None
            self.quant_input = inp_processed

        # If we are executing GPFQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def _get_permutation_list(self, weight: Tensor):
        permutation_list = []
        if self.act_order:
            # We don't need full Hessian, we just need the diagonal
            H_diag = self.quant_input.transpose(2, 1).square().sum(2)
            for group_index in range(self.groups):
                # Re-order Hessian_diagonal so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(H_diag[group_index, :], descending=True)
                perm = perm.to(weight.device)
                permutation_list.append(perm)
        else:
            for group_index in range(self.groups):
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(weight.shape[-1]), device=weight.device)
                permutation_list.append(perm)
        return permutation_list

    def single_layer_update(self, percdamp: float = 0.01):
        assert not self.layer.weight_quant.requires_quant_input, \
            "Error: GPFQ does not support weight quantizers that require quantized inputs."
        weight = self.layer.weight.data
        dev = weight.device
        dtype = weight.dtype
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, SUPPORTED_TCONV_OP):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]

        # stablize H with a dampening factor and then square root the matrix
        norms = torch.zeros((self.groups, self.columns), device=dev, dtype=dtype)
        self.H = self.H.to(dev)
        diag = torch.arange(self.columns, device='cpu')
        for i in range(self.groups):
            damp = percdamp * self.H[i].diag().mean()
            self.H[i, diag, diag] += damp
            norms[i] = self.H[i].diag()  # set the norms post-dampening
            eigvals, eigvecs = torch.linalg.eigh(self.H[i])
            eigvals.clamp_min_(0.0).sqrt_()  # should be positive-definite
            self.H[i] = eigvecs @ torch.diag(eigvals) @ eigvecs.t()
        del eigvecs, eigvals, diag
        self.quant_input = self.H  # NOTE: do this here for the `get_permutation_list` function

        # Try/Except in case the inverse of H cannot be computed
        try:
            self.float_input = self.H.clone()  # going to calculate H^{-1} here
            for i in range(self.groups):
                # from our matrix sqrt, we know G is symmetric and positive-definite, so we
                # can use Cholesky decomposition as an efficient, numerically stable inverse
                L = torch.linalg.cholesky(self.float_input[i])
                self.float_input[i] = torch.cholesky_inverse(L)
            self.float_input = torch.bmm(self.float_input.to(dev), self.G.to(dev))
            del L  # memory management
        except LinAlgError:
            warnings.warn(
                f'Failed to compute the inverse of H for layer {self.name} '
                f'GPFQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H, self.G, self.B  # memory management

        permutation_list = self._get_permutation_list(weight)

        U = torch.zeros(
            weight.shape[0], weight.shape[1], self.float_input.shape[1], device=dev,
            dtype=dtype)  # [Groups, OC/groups, Samples]

        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                i = permutation_list[group_index][t]
                U[group_index] += torch.matmul(
                    weight[group_index, :, i].unsqueeze(1),
                    self.float_input[group_index, :, i].unsqueeze(0),
                )  # [OC/Groups, 1] * [1, INSHAPE[1]]
                norm = norms[group_index, i]
                if norm > 0:
                    q_arg = U[group_index].matmul(self.quant_input[group_index, :, i]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])
                weight[group_index, :, i] = q_arg
            q_groups = self.get_quant_weights(t, 0, permutation_list)
            for group_index in range(self.groups):
                U[group_index] -= torch.matmul(
                    q_groups[group_index].unsqueeze(1),
                    self.quant_input[group_index, :, permutation_list[group_index][t]].unsqueeze(0),
                )

        del self.float_input
        del self.quant_input
