# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import unfoldNd

from brevitas.function import get_upper_bound_on_l1_norm
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import StopFwdException
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
import brevitas.nn as qnn


def random_projection(
        float_input: torch.Tensor, quantized_input: torch.Tensor, compression_rate: float):
    # use random projection to reduce dimensionality
    n = quantized_input.size(1)
    target_dim = int(compression_rate * n)
    dev = float_input.device
    # create gaussian random matrix
    R = torch.normal(mean=0.0, std=1. / math.sqrt(n), size=(target_dim, n), device=dev)
    quantized_input = torch.transpose(quantized_input, 1, 2) @ R.T
    float_input = torch.transpose(float_input, 1, 2) @ R.T
    del R
    # reshape back
    quantized_input = torch.transpose(quantized_input, 1, 2)
    float_input = torch.transpose(float_input, 1, 2)

    return float_input, quantized_input


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
        use_gpfa2q (bool): Whether to use accumulator-aware GPFQ. Default: False
        accumulator_bit_width (Optional, int): The target accumulator bit width. Default: None
        a2q_layer_filter_fnc (Optional, callable): An optional lambda function to filter layers for
            accumulator cosntraints. Should return True for layers to constrain. Default: `lambda x: True`

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
            use_gpfa2q: bool = False,
            accumulator_bit_width: Optional[int] = None,
            a2q_layer_filter_fnc: Optional[Callable[[nn.Module], bool]] = lambda x: True,
            compression_rate: Optional[float] = 0.0,
            collect_float_first: bool = False) -> None:
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

        # GPFA2Q params
        self.use_gpfa2q = use_gpfa2q
        self.accumulator_bit_width = accumulator_bit_width
        self.a2q_layer_filter_fnc = a2q_layer_filter_fnc  # returns true when to use GPFA2Q

        # selecting impl of random proj
        self.compression_rate = compression_rate
        if self.compression_rate < 0.0 or self.compression_rate > 1.0:
            raise ValueError('Compression rate for random projection must be between 0 and 1.')

        # speeding up by collecting float input first so we don't need to do it later
        self.collect_float_first = collect_float_first

    def __enter__(self):
        # initialize gpxq layers
        self.setup_gpxq_layers()
        if self.collect_float_first:
            self.float_collection_hooks = dict()
            # set up hooks for collecting the float input
            for name, layer in self.gpxq_layers.items():
                # Attach float collecting hook
                self.float_collection_hooks[name] = layer.layer.register_forward_hook(
                    layer.collect_float_input)

            # Disable quantization
            self.return_quant_tensor_state = disable_return_quant_tensor(self.model)
            self.disable_quant_inference.disable_param_quantization(self.model, is_training=False)
            self.disable_quant_inference.disable_act_quantization(self.model, is_training=False)

            return self
        else:
            # if we're not collecting, setup original hooks
            return self.setup_gpxq_hooks()

    def finalize_float_collection(self):
        # remove the hooks we attached during the float collection
        for name, hook in self.float_collection_hooks.items():
            hook.remove()

        # Re-enable quantization. If activation quantization is disabled,
        # we also disable bias quantization
        self.disable_quant_inference.enable_param_quantization(self.model, is_training=False)
        if self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(self.model, is_training=False)
        else:
            self.disable_quant_inference.disable_bias_quantization(self.model, is_training=False)
        restore_return_quant_tensor(self.model, self.return_quant_tensor_state)

        # setup the original hooks
        self.setup_gpxq_hooks()

    def catch_stopfwd(self, *args, **kwargs):
        # Collect quant input
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

        if not self.collect_float_first:
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
                self.disable_quant_inference.disable_bias_quantization(
                    self.model, is_training=False)
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
        if (not self.a2q_layer_filter_fnc(layer)) or (not self.use_gpfa2q):
            return GPFQ(
                layer=layer,
                name=name,
                act_order=act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                p=self.p,
                compression_rate=self.compression_rate)
        else:
            return GPFA2Q(
                layer=layer,
                name=name,
                act_order=act_order,
                len_parallel_layers=len_parallel_layers,
                create_weight_orig=create_weight_orig,
                p=self.p,
                accumulator_bit_width=self.accumulator_bit_width,
                compression_rate=self.compression_rate)


class GPFQ(GPxQ):
    """
    Based on https://github.com/YixuanSeanZhou/Quantized_Neural_Nets/tree/main
    """

    def __init__(
            self, layer, name, act_order, len_parallel_layers, create_weight_orig, p,
            compression_rate) -> None:

        super().__init__(layer, name, act_order, len_parallel_layers, create_weight_orig)

        self.float_input = None
        self.quantized_input = None
        self.index_computed = False
        self.p = p
        self.compression_rate = compression_rate

    def collect_float_input(self, module, args, output):
        # this is the hook function to offload the output of this layer to disc
        inp = self.process_input(args)
        batch_size = inp.shape[0]

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, sum(inp.shape[2:])))
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, SUPPORTED_CONV_OP):
            # Pick the correct unfoldNd class
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
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

        inp_processed = inp_processed.cpu()

        if self.float_input is None:
            self.float_input = inp_processed
        else:
            self.float_input = torch.cat([self.float_input, inp_processed], dim=1)

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
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
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
            if self.quantized_input is None:
                self.quantized_input = inp_processed
            else:
                self.quantized_input = torch.cat([self.quantized_input, inp_processed], dim=1)
        # If we are executing GPFQ with group of parallel layers, we keep track of how many forward
        # we executed. Once we executed as many as the number of parallel_layers, we raise
        # StopFwdException
        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self):
        assert not self.layer.weight_quant.requires_quant_input, "Error: GPFQ does not support weight quantizers that require quantized inputs."
        weight = self.layer.weight.data
        dev = weight.device
        dtype = weight.dtype
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(
                    self.layer,
                (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d, qnn.QuantConvTranspose3d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]
        if self.compression_rate > 0.0:
            self.float_input, self.quantized_input = random_projection(self.float_input, self.quantized_input, self.compression_rate)
        self.float_input = self.float_input.to(dev)
        self.quantized_input = self.quantized_input.to(dev)
        U = torch.zeros(
            weight.shape[0], weight.shape[1], self.float_input.shape[1], device=dev, dtype=dtype)
        # We don't need full Hessian, we just need the diagonal
        self.H_diag = self.quantized_input.transpose(2, 1).square().sum(
            2)  # summing over Batch dimension
        permutation_list = []
        for group_index in range(self.groups):
            if self.act_order:
                # Re-order Hessian_diagonal so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(self.H_diag[group_index, :], descending=True)
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(weight.shape[-1]), device=dev)
            permutation_list.append(perm)
        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                U[group_index] += torch.matmul(
                    weight[group_index, :, permutation_list[group_index][t]].unsqueeze(1),
                    self.float_input[group_index, :, permutation_list[group_index][t]].unsqueeze(
                        0))  #[OC/Groups, 1] * [1, INSHAPE[1]]
                norm = torch.linalg.norm(
                    self.quantized_input[group_index, :, permutation_list[group_index][t]], 2) ** 2
                if norm > 0:
                    q_arg = U[group_index].matmul(
                        self.quantized_input[group_index, :,
                                             permutation_list[group_index][t]]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])

                weight[group_index, :, permutation_list[group_index][t]] = q_arg
            q = self.get_quant_weights(t, 0, permutation_list)
            for group_index in range(self.groups):
                U[group_index] -= torch.matmul(
                    q[group_index].unsqueeze(1),
                    self.quantized_input[group_index, :,
                                         permutation_list[group_index][t]].unsqueeze(0))

        del self.float_input
        del self.quantized_input


class GPFA2Q(GPFQ):

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            accumulator_bit_width,
            p,
            compression_rate) -> None:
        GPFQ.__init__(
            self,
            layer=layer,
            name=name,
            act_order=act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            p=p,
            compression_rate=compression_rate)
        self.accumulator_bit_width = accumulator_bit_width
        assert self.accumulator_bit_width is not None

    def single_layer_update(self):
        # raise error in case no quant-input is here
        if self.quant_metadata is None:
            raise ValueError('Expected self.quant_metadata to calculate L1-norm upper bound, but recevied None. ' + \
                'Make sure that either the input to the model is a IntQuantTensor or the layer has an input quant enabled. ' \
                'Also, check if `use_quant_activations=True` in `gpfq_mode` when `accumulator_bit_width` is specified. ' + \
                'Alternatively, provide a custom `a2q_layer_filter_fnc` to `gpfq_mode` to filter layers without a quant_tensor input.')
        weight = self.layer.weight.data
        dev = weight.device
        dtype = weight.dtype
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if isinstance(self.layer, (qnn.QuantConvTranspose1d, qnn.QuantConvTranspose2d)):
                weight = weight.transpose(1, 0)  # This performs a view
            weight = weight.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])  # [Groups, OC/Groups, IC]
        U = torch.zeros(
            weight.shape[0], weight.shape[1], self.float_input.shape[1], device=dev, dtype=dtype)
        if self.compression_rate > 0.0:
            self.float_input, self.quantized_input = random_projection(self.float_input, self.quantized_input, self.compression_rate)
        self.float_input = self.float_input.to(dev)
        self.quantized_input = self.quantized_input.to(dev)

        # get upper bound
        input_bit_width = self.quant_metadata.bit_width
        input_is_signed = self.quant_metadata.signed
        T = get_upper_bound_on_l1_norm(
            torch.tensor(self.accumulator_bit_width), input_bit_width, input_is_signed)
        s = self.layer.weight_quant.scale()
        if s.ndim > 1:
            s = s.view(self.groups, -1)  # [Groups, OC/Groups]

        # initialize cumulative l1-norm
        z = torch.zeros(weight.shape[:-1], device=dev)

        # We don't need full Hessian, we just need the diagonal
        self.H_diag = self.quantized_input.transpose(2, 1).square().sum(
            2)  # summing over Batch dimension
        permutation_list = []
        for group_index in range(self.groups):
            if self.act_order:
                # Re-order Hessian_diagonal so that weights associated to
                # higher magnitude activations are quantized first
                perm = torch.argsort(self.H_diag[group_index, :], descending=True)
            else:
                # No permutation, permutation tensor is a ordered index
                perm = torch.tensor(range(weight.shape[-1]), device=dev)
            permutation_list.append(perm)

        for t in range(weight.shape[-1]):
            for group_index in range(self.groups):
                U[group_index] += torch.matmul(
                    weight[group_index, :, permutation_list[group_index][t]].unsqueeze(1),
                    self.float_input[group_index, :, permutation_list[group_index][t]].unsqueeze(
                        0))  #[OC/Groups, 1] * [1, INSHAPE[1]]
                norm = torch.linalg.norm(
                    self.quantized_input[group_index, :, permutation_list[group_index][t]], 2) ** 2
                if norm > 0:
                    q_arg = U[group_index].matmul(
                        self.quantized_input[group_index, :,
                                             permutation_list[group_index][t]]) / norm
                else:
                    q_arg = torch.zeros_like(U[group_index, :, 0])

                max_q_arg = s * torch.clamp_min(T - z, 0.)
                q_arg = q_arg.sign() * torch.clamp_max(q_arg.abs(), max_q_arg[group_index, :])
                weight[group_index, :, permutation_list[group_index][t]] = q_arg
            q = self.get_quant_weights(t, 0, permutation_list)
            z += q.abs() / s  # increment cumulative l1-norm

            for group_index in range(self.groups):
                U[group_index] -= torch.matmul(
                    q[group_index].unsqueeze(1),
                    self.quantized_input[group_index, :,
                                         permutation_list[group_index][t]].unsqueeze(0))

        del self.float_input
        del self.quantized_input
