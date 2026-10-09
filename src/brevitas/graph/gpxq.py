# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
import math
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import torch
import torch.nn as nn
import unfoldNd

from brevitas.graph.layerwise_hook import LayerHandler
from brevitas.graph.layerwise_hook import layerwise_hook_mode
from brevitas.graph.utils import get_batch_dim
from brevitas.graph.utils import is_conv_transposed
from brevitas.graph.utils import is_quant_module
import brevitas.nn as qnn
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.torch_utils import rename_tensor

SUPPORTED_CONV_OP = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def process_layer_input(
        layer: nn.Module,
        inp,
        groups: int,
        quant_metadata: Optional[object] = None) -> Tuple[torch.Tensor, Optional[object]]:
    """Preprocess a layer's input into the [groups, in_features, batch] layout used
    to accumulate the Hessian, shared by GPxQ and PiSO.

    quant_metadata is threaded in/out (rather than mutated on an instance) so
    the helper can stay a free function: if the input is a quantized activation and
    quant_metadata is still None, it is captured and returned so a caller that
    needs quant-input-dependent weight quantization can reuse it. Callers that do
    not need it (PiSO) can ignore the returned value.

    Returns (inp_processed, quant_metadata).
    """
    # Input is a tuple, so we take first element
    inp = inp[0]
    if is_quant_module(layer):
        inp = layer.input_quant(inp)
        is_quant_enabled = layer.weight_quant.is_quant_enabled
    else:
        is_quant_enabled = False

    # If using quantized activations, inp could be QuantTensor. In
    # this case, we overwrite the metadata.
    if isinstance(inp, QuantTensor):
        if is_quant_enabled and quant_metadata is None:
            quant_metadata = layer.input_quant.cache_class(inp, metadata_only=True)
        inp = inp.value

    # If input is unbatched, add batch_size = 1
    if len(inp.shape) == 1:
        warnings.warn("Found unbatched input, adding batch dimension equal to 1")
        inp = inp.unsqueeze(0)

    # Define batch size before re-organizing the input. Prefer batch_dim/batch_first exposed
    # by the module; fall back to named tensors (PyTorch < 2.13).
    batch_dim = get_batch_dim(layer, inp)
    # Strip any legacy dimension names before reshaping (no-op on PyTorch >= 2.13).
    inp = rename_tensor(inp, None)
    if batch_dim:
        inp = inp.transpose(0, batch_dim)

    # Preprocess the input to compute the Hessian
    if isinstance(layer, nn.Linear):
        if len(inp.shape) > 2:
            inp = inp.reshape((-1, sum(inp.shape[2:])))
        inp = inp.t()
        # For QuantLinear layer, groups will be 1
        inp_processed = inp.unsqueeze(0)

    if isinstance(layer, SUPPORTED_CONV_OP):
        # Pick the correct unfoldNd class
        if is_conv_transposed(layer):
            unfold_impl = unfoldNd.UnfoldTransposeNd
        else:
            unfold_impl = unfoldNd.UnfoldNd

        unfold = unfold_impl(
            layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride)

        # Split input based on how many groups in convolution
        inp_by_group = torch.chunk(inp, groups, 1)
        inp_processed = []
        # Preprocess input by group
        for i, inp in enumerate(inp_by_group):
            inp = unfold(inp)
            inp = inp.transpose(1, 0)
            inp = inp.flatten(1)
            inp_processed.append(inp)
        inp_processed = torch.stack(inp_processed)

    return inp_processed, quant_metadata


class gpxq_mode(layerwise_hook_mode):
    """
    Apply GPxQ algorithm.

    Args:
        model (Module): The model to quantize with GPxQ
        group_of_parallel_layers (Optional, List[str]): .List of lists where each inner list is a group
            of layer names that can be optimized in parallel. Default: None
        inplace (bool): Wheter to apply GPFQ inplace or perform a deepcopy. Default: True
        create_weight_orig (bool): If True, store the original floating point weights before applying
            gpxq. These weights will be used anytime quantization is disabled. Default: True
        use_quant_activations (bool): Wheter to leave quantize activations enabled while performing
            GPxQ. Default: False
        act_order (bool): Whether to order greedy path following by Hessian approximation. Default: False
        return_forward_output (bool): If True, returns the output of the forward pass. Otherwise the
            forward call inside the context manager returns None. Default: False
        device (str): Device the buffers are stored on. Default: cpu
        dtype (torch.dtype): Datatype the buffers are stored in. Default: torch.float32

    Example:
        >>> with torch.no_grad():
        >>>     with gpxq_mode(model) as gpxq:
        >>>         gpxq_mode = gpxq.model
        >>>         for i in tqdm(range(gpxq.num_layers)):
        >>>             for img, t in calib_loader:
        >>>                 img = img.cuda()
        >>>                 gpxq_mode(img)
        >>>             gpxq.update()
    """

    def __init__(
            self,
            model,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            act_order: bool = False,
            return_forward_output: bool = False,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32) -> None:
        if not inplace:
            model = deepcopy(model)
        # Note that if use_quant_activations = True, the super() context manager
        # is equivalent to a nullcontext
        super().__init__(
            model=model,
            disable_act_quant=not use_quant_activations,
            disable_bias_quant=not use_quant_activations,
            create_weight_orig=create_weight_orig,
            group_of_parallel_layers=group_of_parallel_layers,
        )
        self.use_quant_activations = use_quant_activations
        # Quantize following magnitude of activation
        self.act_order = act_order
        # the device and dtype of the buffers
        self.device = device
        self.dtype = dtype

        self.return_forward_output = return_forward_output

    def _is_module_supported(self, module):
        if is_quant_module(module):
            is_quant_enabled = module.weight_quant.is_quant_enabled
        else:
            is_quant_enabled = False
        if isinstance(module, (nn.Linear, *SUPPORTED_CONV_OP)):
            # ConvTranspose is temporarily unsupported in GPxQ
            # See https://github.com/Xilinx/brevitas/issues/1479
            if is_conv_transposed(module):
                warnings.warn("ConvTranspose is temporarily unsupported for GPxQ, skipping.")
                return False
            return is_quant_enabled
        else:
            return False


class GPxQ(ABC):

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers=1,
            create_weight_orig=True,
            device='cpu',
            dtype=torch.float32) -> None:
        self.layer = layer
        self.name = name
        self.act_order = act_order
        self.create_weight_orig = create_weight_orig
        # device and dtype of buffers; 'same' means using the same device for the buffer as the layer weights
        self.device = layer.weight.device if device == 'same' else device
        self.dtype = dtype

        weight_shape = torch.tensor(layer.weight.shape)

        if create_weight_orig and not hasattr(self.layer, 'weight_orig'):
            self.layer.register_buffer('weight_orig', layer.weight.detach().clone().cpu())

        # By default, use groups = 1
        self.groups = 1
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight_shape[1], weight_shape[0] = weight_shape[0], weight_shape[1]
            self.groups = self.layer.groups

        # Number of rows is equal to the output channels (OC)
        self.rows = weight_shape[0]
        # Number of columns is equal to the input channels (IC)
        self.columns = torch.prod(weight_shape[1:])
        self.len_parallel_layers = len_parallel_layers

        self.disable_pre_forward_hook = False
        # Some layers require knowledge from quant inputs to compute quant weights
        self.quant_metadata = None

    @property
    def use_intermediate_buffer(self):
        # By default, we are optimizing for minimizing peak memory usage, which is
        # when self.device=='cpu'. Since the compute is done on the GPU but the buffers
        # are on the GPU, we optimize the CPU to GPU transfer using in-place copy to
        # pinned memory in an intermediate buffer, usually self.B
        return self.device == 'cpu'

    def process_input(self, inp):
        inp_processed, self.quant_metadata = process_layer_input(
            self.layer, inp, self.groups, self.quant_metadata)
        return inp_processed

    @abstractmethod
    def update_batch(self):
        pass

    @abstractmethod
    def single_layer_update(self):
        pass

    # -------------------------------------------------------------------------
    # Extension hooks.
    #
    # These provide the default (plain GPxQ) behaviour and are overridden by
    # mixins to plug in group-aware permutations (GroupAwarePermutationMixin) or
    # PiSO scale optimization (PiSOMixin, see brevitas.graph.piso). GPTQ/Qronos
    # call them so their own bodies stay free of algorithm-selection branching.
    # -------------------------------------------------------------------------

    def _resolve_blocksize(self, num_blocks: int) -> int:
        """Number of columns processed per error-correction block.

        Default: split the columns into num_blocks blocks (plain GPxQ).
        PiSO overrides this to align blocks with quantization groups.
        """
        return math.ceil(self.columns / num_blocks)

    def _act_order_permutation(self, hessian_group: torch.Tensor) -> torch.Tensor:
        """Column ordering used when act_order is enabled, for one group.

        Only called from the act_order branch of the algorithms. Default:
        descending diagonal-Hessian order. GroupAwarePermutationMixin overrides
        this to keep quantization groups contiguous.
        """
        return torch.argsort(torch.diag(hessian_group), descending=True)

    def _on_error_correction_starts(self, perm: torch.Tensor) -> None:
        """Called once the (damped) Hessian is ready, before error correction.

        No-op by default; PiSO uses it to optimize the layer scale on the fixed
        grid before weights are error-corrected.
        """
        pass

    def _on_column_reached(self, column_idx: int) -> None:
        """Signal that error correction is about to process column column_idx.

        Called once per column. No-op by default; PiSO uses it to optimize a
        group's scale when column_idx starts a new group, before its weights
        are quantized.
        """
        pass

    def _on_layer_finished(self) -> None:
        """Called after the layer is fully processed. No-op by default.

        A teardown hook (e.g. PiSO releases its per-layer state here); also
        invoked on the error path when the layer is skipped.
        """
        pass

    def get_quant_weights(self, i, i1, permutation_list, with_quant_history=False):

        # If the weight quantizer has not been initialized, raise an error
        for m in self.layer.weight_quant.modules():
            if hasattr(m, 'init_done') and not m.init_done:
                raise RuntimeError(
                    "Weight quantizer not initialized. Run a forward pass after quantization and try again."
                )

        # We need to recompute quant weights at runtime since our float weights are being updated
        # Add offset in case of blockwise computation
        i = i1 + i

        # For QuantLinear and for some QuantConvolutional layers, we exploit the possibility
        # of quantizing only a subset of the entire matrix speeding up the computation of GPxQ
        no_slice = False
        # Groupwise Quantization does not support slicing
        no_slice = no_slice or self.layer.weight_quant.is_groupwise
        # If we need quantization of past channels, we do not use slicing
        no_slice = no_slice or with_quant_history
        # If we are in export mode (i.e., inference mode), we do not slice for torch.compile
        # compatibility
        no_slice = no_slice or self.layer.weight_quant.export_mode

        if isinstance(self.layer, qnn.QuantLinear):
            if no_slice:

                # No slicing, not optimized
                q = self.layer.quant_weight(quant_input=self.quant_metadata)
                q = _unpack_quant_tensor(q).unsqueeze(0)  # [1, OC, IC]
                if with_quant_history:
                    return q[:, :, permutation_list[0][:i]]  # [1, OC, i]
                index = permutation_list[0][i]  # only 1 group for linear layers
                q = q[:, :, index:index + 1]  # [1, OC, 1]
            else:
                index = permutation_list[0][i]
                subtensor_slice_list = [None, (index, index + 1)]
                q = _unpack_quant_tensor(
                    self.layer.quant_weight(
                        subtensor_slice_list=subtensor_slice_list,
                        quant_input=self.quant_metadata)).unsqueeze(0)  # [1, OC, 1]
        elif isinstance(self.layer, SUPPORTED_CONV_OP):
            # Depthwise and ConvTranspose does not support slicing
            no_slice_conv = no_slice or (self.groups > 1 or is_conv_transposed(self.layer))

            if no_slice_conv:

                quant_weight = self.layer.quant_weight(quant_input=self.quant_metadata)
                quant_weight = _unpack_quant_tensor(quant_weight)

                if is_conv_transposed(self.layer):
                    quant_weight = quant_weight.transpose(1, 0)  # This performs a view
                quant_weight = quant_weight.flatten(1)
                quant_weight = quant_weight.view(self.groups, -1, quant_weight.shape[-1])

                if self.act_order:
                    for ii, perm in enumerate(permutation_list):
                        quant_weight[ii, :, :] = quant_weight[ii, :, perm]

                if with_quant_history:
                    return quant_weight[:, :, :i]  # [groups, OC/groups, i]
                q = quant_weight[:, :, i:i + 1]  # [groups, OC/groups, 1]
            else:
                index = permutation_list[0][i]
                shapes = self.layer.weight.shape[1:]
                index_2d_to_nd = []
                residual_index = index.item()
                for shape in shapes[::-1]:
                    index_2d_to_nd.append((residual_index % shape, residual_index % shape + 1))
                    residual_index = residual_index // shape
                index_2d_to_nd = index_2d_to_nd[::-1]
                index_2d_to_nd.insert(0, None)
                q = _unpack_quant_tensor(
                    self.layer.quant_weight(
                        subtensor_slice_list=index_2d_to_nd,
                        quant_input=self.quant_metadata)).flatten(1)  # [OC, 1]
                q = q.unsqueeze(0)  # [1, OC, 1]
        # We need to remove the last dim
        q = q.squeeze(2)  # [groups, OC/groups] or [1, OC]
        return q
