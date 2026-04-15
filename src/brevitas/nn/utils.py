# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import ScalingImplType
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.utils.torch_utils import compute_channel_view_shape


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias):
    denom = torch.sqrt(bn_var + bn_eps)
    mul_factor = bn_weight / denom
    add_factor = -bn_mean * mul_factor + bn_bias
    return mul_factor, add_factor


def merge_bn(layer, bn, output_channel_dim=0):
    from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjectorBase
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
    out = mul_add_from_bn(
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_eps=bn.eps,
        bn_weight=bn.weight.data.clone(),
        bn_bias=bn.bias.data.clone())
    mul_factor, add_factor = out
    out_ch_weight_shape = compute_channel_view_shape(layer.weight, output_channel_dim)
    layer.weight.data.mul_(mul_factor.view(out_ch_weight_shape))
    if layer.bias is not None:
        out_ch_bias_shape = compute_channel_view_shape(layer.bias, channel_dim=0)
        layer.bias.data.mul_(mul_factor.view(out_ch_bias_shape))
        layer.bias.data.add_(add_factor.view(out_ch_bias_shape))
    else:
        layer.bias = Parameter(add_factor)
    if (hasattr(layer, 'weight_quant') and
            isinstance(layer.weight_quant, WeightQuantProxyFromInjectorBase)):
        layer.weight_quant.init_tensor_quant()
    if (hasattr(layer, 'bias_quant') and
            isinstance(layer.bias_quant, BiasQuantProxyFromInjectorBase)):
        layer.bias_quant.init_tensor_quant()


def rename_state_dict_by_prefix(old_prefix, new_prefix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)


def rename_state_dict_by_postfix(old_postfix, new_postfix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.endswith(old_postfix):
            new_key = k[:len(k) - len(old_postfix)] + new_postfix
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)


class merge_quant_weights:
    """Context manager that merges quantized weights into model weights.

    This could be useful for example with Learned Round.
    After learned round training, the rounding decision for each weight element is
    deterministic. This context manager uses forward hooks to discover the association
    between weight tensors and its quantized counterparts, and update the module's weights.

    Usage::

        model.eval()
        with merge_learned_round(model):
            model(sample_input)
        # Weights are now merged and rounding mode is ROUND.

    Args:
        model: A model containing quantised layers with learned round quantisers.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            disable_quant: bool = True,
            preserve_original_weights: bool = False) -> None:
        self._model = model
        self._hooks: List[RemovableHandle] = []
        self._module_tensor_id_mapping = {}
        self.disable_quant = disable_quant
        self.hook_check = False
        self.preserve_original_weights = preserve_original_weights

    def __enter__(self) -> 'merge_quant_weights':
        # Imported here to avoid a circular import
        from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase

        def model_hook(module, args, output):
            if self.hook_check:
                raise RuntimeError(
                    "Calling multiple forward pass within the context manager is not supported")
            self.hook_check = True

        def hook(module, args, output):
            input_tensor = args[0]
            with torch.no_grad():
                for m in module.tracked_module_list:
                    # We match the module based on its weights and the ID of the tensor to quantize
                    if id(m.weight.data) == id(input_tensor.data):
                        if self.preserve_original_weights:
                            m.weight_orig = m.weight.data.cpu()
                        m.weight.data = output.value.data
                        # We track how many modules have been converted
                        if module not in self._module_tensor_id_mapping:
                            self._module_tensor_id_mapping[module] = [id(m.weight.data)]
                        else:
                            self._module_tensor_id_mapping[module].append(id(m.weight.data))

        # Register Proxy hooks
        for module in self._model.modules():
            if not isinstance(module, WeightQuantProxyFromInjectorBase):
                continue
            self.change_scale_impl_type(module)
            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)

        # Register Model hook
        handle = self._model.register_forward_hook(model_hook)
        self._hooks.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        with torch.no_grad():
            for module in self._module_tensor_id_mapping:
                self._reset_quantizer(module)

    def change_scale_impl_type(self, proxy) -> None:
        reinit_on_state_dict = config.REINIT_ON_STATE_DICT_LOAD
        ignore_missing_key = config.IGNORE_MISSING_KEYS
        config.REINIT_ON_STATE_DICT_LOAD = False
        config.IGNORE_MISSING_KEYS = True
        state_dict = proxy.state_dict()
        proxy.quant_injector = proxy.quant_injector.let(
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS)
        proxy.init_tensor_quant()
        proxy.load_state_dict(state_dict, strict=False)
        config.IGNORE_MISSING_KEYS = ignore_missing_key
        config.REINIT_ON_STATE_DICT_LOAD = reinit_on_state_dict

    @staticmethod
    def _reset_quantizer(proxy) -> None:
        """Switch a weight quant proxy from LearnedRound back to standard Round."""
        reinit_on_state_dict = config.REINIT_ON_STATE_DICT_LOAD
        ignore_missing_key = config.IGNORE_MISSING_KEYS
        config.REINIT_ON_STATE_DICT_LOAD = False
        config.IGNORE_MISSING_KEYS = True
        state_dict = {
            k: v for k,
            v in proxy.state_dict().items() if not k.endswith('float_to_int_impl.value')}

        proxy.quant_injector = proxy.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.ROUND)
        proxy.init_tensor_quant()
        proxy.load_state_dict(state_dict, strict=False)
        config.IGNORE_MISSING_KEYS = ignore_missing_key
        config.REINIT_ON_STATE_DICT_LOAD = reinit_on_state_dict


def check_tensors_same_ptr(tensor_list):
    pointers = []
    for t in tensor_list:
        if hasattr(t, 'data_ptr'):
            ptr = t.data_ptr()
            pointers.append(ptr)
        elif hasattr(t, 'value') and hasattr(t.value, 'data_ptr'):
            pointers.append(t.value.data_ptr())
        else:
            return False
    return all(p == pointers[0] for p in pointers)
