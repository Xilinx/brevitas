from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math
import warnings

import torch
from torch.linalg import LinAlgError
import torch.nn as nn

from brevitas.graph.calibrate import DisableEnableQuantization
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor


class StopFwdException(Exception):
    pass


@dataclass
class LayerHandler:
    layer_name: str = None


class gptq_mode:
    """
    Apply GPTQ algorithm https://arxiv.org/abs/2210.17323.

    Args:
        model (Module): The model to quantize with GPTQ
        inplace (bool): Wheter to apply GPTQ inplace or perform a deepcopy. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPTQ. Default: False

    Example:
        >>> with torch.no_grad():
        >>>     with gptq_mode(model) as gptq:
        >>>         gptq_model = gptq.model
        >>>         for i in tqdm(range(gptq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gptq_model(img)
        >>>             gptq.update()
    """

    def __init__(
            self,
            model,
            inplace=True,
            use_quant_activations=True,
            num_blocks=4,
            act_order=False) -> None:
        if not inplace:
            model = deepcopy(model)
        self.model = model
        self.use_quant_activations = use_quant_activations
        self.hook_dict = dict()
        self.gptq_layers = dict()
        # reference for each layer to update
        self.current_layer = LayerHandler()
        # How many layer to optimize
        self.num_layers = 0
        # Quantize following magnitude of activation
        self.act_order = act_order
        # How many subblock to use during GPTQ for each layer
        self.num_blocks = num_blocks
        self.disable_quant_inference = DisableEnableQuantization()
        self.orig_forward = self.model.forward
        self.model.forward = self.catch_stopfwd

    def _is_module_supported(self, module):
        if isinstance(module, qnn.QuantConv2d) and (module.groups == 1 or
                                                    (module.groups == module.out_channels)):
            return True
        elif isinstance(module, qnn.QuantLinear):
            return True
        else:
            return False

    def __enter__(self):
        # Print warning if hooks are attached to any module, since the normal forward flow of the
        # network is highly disrupted during GPTQ
        for name, module in self.model.named_modules():
            if len(module._forward_hooks) > 0 or len(module._forward_pre_hooks):
                warnings.warn(
                    f'Hooks detected during setup for GPTQ. '
                    f'Behaviour might deviate from what expected.')

            # Attach hooks for GPTQ
            if self._is_module_supported(module):
                gptq = GPTQ(module, name, num_blocks=self.num_blocks, act_order=self.act_order)
                hook_fn = partial(gptq.update_batch, current_layer=self.current_layer)
                self.hook_dict[name] = module.register_forward_hook(hook_fn)
                self.gptq_layers[name] = gptq
        if not self.use_quant_activations:
            self.disable_quant_inference.disable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.disable_bias_quantization(
                self.model, is_training=self.model.training)
        self.num_layers = len(self.gptq_layers)
        return self

    def __exit__(self, type, value, traceback):
        self.model.forward = self.orig_forward
        if not self.use_quant_activations:
            self.disable_quant_inference.enable_act_quantization(
                self.model, is_training=self.model.training)
            self.disable_quant_inference.enable_bias_quantization(
                self.model, is_training=self.model.training)

    def update(self):
        self.gptq_layers[self.current_layer.layer_name].single_layer_update()
        self.hook_dict[self.current_layer.layer_name].remove()

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass


class GPTQ():
    """
    Adapted from https://github.com/IST-DASLab/gptq, released under the following LICENSE:

    Copyright 2023 IST-DASLab

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def __init__(self, layer, name, num_blocks, act_order) -> None:
        self.layer = layer
        self.name = name
        self.num_blocks = num_blocks
        self.act_order = act_order

        weight = layer.weight.data
        dev = weight.device

        self.groups = 1
        if isinstance(self.layer, qnn.QuantConv2d):
            weight = weight.flatten(1)
            self.groups = self.layer.groups
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.H = torch.zeros((self.groups, self.columns, self.columns), device=dev)
        self.nsamples = 0

    def update_batch(self, module, input, out, current_layer):
        # Update reference to current layer
        current_layer.layer_name = self.name

        # Input is a tuple, so we take first element
        inp = input[0]
        # If using Quant Activations, inp could be QuantTensor
        if isinstance(inp, QuantTensor):
            inp = inp.value

        # If input is unbatched, add batch_size = 1
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        # Define batch size before re-organizing the input
        batch_size = inp.shape[0]

        # Preprocess the input to compute the Hessian
        if isinstance(self.layer, qnn.QuantLinear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            # For QuantLinear layer, groups will be 1
            inp_processed = inp.unsqueeze(0)

        if isinstance(self.layer, qnn.QuantConv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride)

            # Split input based on how many groups in convolution
            inp_by_group = torch.chunk(inp, self.groups, 1)
            inp_processed = []
            # Preprocess input by group
            for i, inp in enumerate(inp_by_group):
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
                inp_processed.append(inp)
            inp_processed = torch.stack(inp_processed)

        # Hessian computation
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.float()
        self.H += inp_processed.bmm(inp_processed.transpose(2, 1))
        raise StopFwdException

    def single_layer_update(self, percdamp=.01):
        weight = self.layer.weight.data
        dev = weight.device
        if isinstance(self.layer, qnn.QuantConv2d):
            weight = weight.flatten(1)

        blocksize = math.ceil(weight.shape[1] / self.num_blocks)

        permutation_list = []
        if self.groups > 1:
            for i in range(self.groups):
                dead = torch.diag(self.H[i, :, :]) == 0
                self.H[i, dead, dead] = 1
                weight[i, dead] = 0
                if self.act_order:
                    perm = torch.argsort(torch.diag(self.H[i, :, :]), descending=True)
                    self.H[i, :, :] = self.H[i, perm, :][:, perm]
                else:
                    # No permutation
                    perm = list(range(self.H.shape[-1]))
                permutation_list.append(perm)
        else:
            dead = torch.diag(self.H[0, :, :]) == 0
            self.H[0, dead, dead] = 1
            weight[:, dead] = 0
            if self.act_order:
                perm = torch.argsort(torch.diag(self.H[0, :, :]), descending=True)
                self.H = self.H[:, perm, :][:, :, perm]
            else:
                # No permutation
                perm = list(range(self.H.shape[-1]))
            permutation_list.append(perm)

        # Try/Except in case the inverse Hessian cannot be computed
        try:
            for i in range(self.groups):
                damp = percdamp * torch.mean(torch.diag(self.H[i, :, :]))
                diag = torch.arange(self.columns, device=dev)
                self.H[i, diag, diag] += damp
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :])
                self.H[i, :, :] = torch.cholesky_inverse(self.H[i, :, :])
                self.H[i, :, :] = torch.linalg.cholesky(self.H[i, :, :], upper=True)
            Hinv = self.H
        except LinAlgError as e:
            warnings.warn(
                f'Failed to compute the inverse of the Hessian for layer {self.name} '
                f'GPTQ will not be applied. '
                f'Increasing the number of samples might fix this issue')
            return
        finally:
            del self.H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            if len(permutation_list) == 1:
                perm = permutation_list[0]
                weight1 = weight[:, perm[i1:i2]]
            else:
                weight1 = torch.empty(weight.shape[0], count, device=dev)
                for ii, perm in enumerate(permutation_list):
                    weight1[ii, :] = weight[ii, perm[i1:i2]]

            Err1 = torch.zeros_like(weight1)
            Hinv1 = Hinv[:, i1:i2, i1:i2]
            for i in range(count):
                w = weight1[:, i]
                d = Hinv1[:, i, i]
                q = self.get_quant_weights(i, i1, i2, permutation_list)

                err1 = (w - q) / d
                if self.groups > 1:
                    # In case of depthwise convs, each weight matrix interacts with only
                    # part of the input values, thus with only one of the hessian matrix
                    for ii in range(self.groups):
                        weight1[ii, i:] -= err1[ii] * Hinv1[ii, i, i:]
                else:
                    weight1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[0, i, i:].unsqueeze(0))
                Err1[:, i] = err1

                # We need to update the original weights
                weight[:, perm[i1:i2][i:]] = weight1[:, i:]

            if self.groups > 1:
                # In case of depthwise convs, each weight matrix interacts with only
                # part of the input values, thus with only one of the hessian matrix
                for ii, perm in enumerate(permutation_list):
                    weight[ii:ii + 1, perm[i2:]] -= Err1[ii:ii + 1, :].matmul(Hinv[ii, i1:i2, i2:])
            else:
                perm = permutation_list[0]
                weight[:, perm[i2:]] -= Err1.matmul(Hinv[0, i1:i2, i2:])

    def get_quant_weights(self, i, i1, i2, permutation_list):
        quant_weight = self.layer.quant_weight()
        quant_weight = quant_weight.value
        if isinstance(self.layer, qnn.QuantConv2d):
            quant_weight = quant_weight.flatten(1)

        if self.act_order:
            # If act order is enabled, permute quant weight to match the float32 counterpart
            if len(permutation_list) == 1:
                quant_weight = quant_weight[:, permutation_list[0]]
            else:
                for ii, perm in enumerate(permutation_list):
                    quant_weight[ii, :] = quant_weight[ii, perm]

        quant_weight1 = quant_weight[:, i1:i2]
        q = quant_weight1[:, i]
        return q
