# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import sys
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.nn import QuantHardTanh
from brevitas.nn import QuantLinear
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import ParameterQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.proxy.quant_proxy import QuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas.proxy.runtime_quant import ClampQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor.base_quant_tensor import _unpack_quant_tensor

from .base import Transform

__all__ = [
    'ClipFloatWeights',
    'QuantizationStatusManager',
    'quantization_status_manager',
    'bias_correction_mode',
    'calibration_mode',
    'load_quant_model_mode']

_PARAM_PROXIES = (WeightQuantProxyFromInjectorBase, BiasQuantProxyFromInjectorBase)

_WEIGHT_PROXIES = (WeightQuantProxyFromInjectorBase,)
_BIAS_PROXIES = (BiasQuantProxyFromInjectorBase,)

_ACC_PROXIES = (TruncQuantProxyFromInjector, ClampQuantProxyFromInjector)

_LAYERS_TO_CLIP = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d)

BN_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def unpack_input(inp: Union[Tuple[QuantTensor], QuantTensor]) -> torch.Tensor:
    inp = _unpack_quant_tensor(inp)
    # Unpack element in the collection
    if isinstance(inp, (tuple, list, dict)):
        assert len(inp) == 1, "A collection with a single element is expected"
        # Return the single element in the iterable
        if isinstance(inp, (tuple, list)):
            return inp[0]
        else:
            return next(iter(inp.values()))

    return inp


def extend_collect_stats_steps(module: nn.Module) -> None:
    if hasattr(module, 'collect_stats_steps'):
        # We extend the collect steps in PTQ to match potentially long calibrations
        module.collect_stats_steps = sys.maxsize - 1


def set_collect_stats_to_average(module: nn.Module) -> None:
    if hasattr(module, 'collect_stats_steps') and hasattr(module, 'momentum'):
        # Set the average for stats collection to a true average (default is EMA)
        module.momentum = None


def finalize_collect_stats(module: nn.Module) -> None:
    if hasattr(module, 'init_scale'):
        module.init_scale()
    elif hasattr(module, 'init_zp'):
        module.init_zp()


class bias_correction_mode:

    def __init__(self, model: nn.Module, enabled: bool = True, skip_if_no_bias: bool = False):
        self.model = model
        self.bias_correction = _BiasCorrection(skip_if_no_bias=skip_if_no_bias)
        self.enabled = enabled
        self.hooks = []
        self.output_quant_modules = []

    def __enter__(self):
        if self.enabled:
            for module in self.model.modules():
                # Disable output quant so that the bias correction can be merged in the bias
                if hasattr(module, 'output_quant') and module.output_quant.is_quant_enabled:
                    QuantizationStatusManager.disable_act_quantization(
                        module.output_quant, is_training=False)
                    self.output_quant_modules.append(module)
            self.bias_correction.register_hook_to_wbiol(self.model, self.hooks)

    def __exit__(self, type, value, traceback):
        self.bias_correction.apply_correction(self.model)
        for module in self.output_quant_modules:
            # Re-enable output quantization
            QuantizationStatusManager.enable_act_quantization(
                module.output_quant, is_training=False)
        for hook in self.hooks:
            hook.remove()


class load_quant_model_mode:

    def __init__(self, model):
        self.model = model
        self.tracked_modules = []

    def __enter__(self):
        for module in self.model.modules():
            if issubclass(type(module), QuantWBIOL):
                module._quant_load_model_mode = True

    def __exit__(self, *args, **kwargs):
        for module in self.model.modules():
            if issubclass(type(module), QuantWBIOL):
                module._quant_load_model_mode = False


class ClipFloatWeights(Transform):

    def __init__(self, threshold=15., layers_to_clip=_LAYERS_TO_CLIP) -> None:
        super(ClipFloatWeights, self).__init__()
        self.threshold = threshold
        self.layers_to_clip = layers_to_clip

    def apply(self, model):
        for module in model.modules():
            if isinstance(module, self.layers_to_clip):
                module.weight.data.clamp_(-self.threshold, self.threshold)
        return model


class QuantizationStatusManager:

    @staticmethod
    def _set_act_quantization(
            model: nn.Module,
            disable_quant: bool,
            is_training: bool,
            call_act_quantizer_impl: bool = False) -> Dict[nn.Module, bool]:
        # Save previous state of activation quantizers
        previous_state = {}
        previous_observer_state = {}
        # If call_act_quantizer_impl is set to True, the quantization will be performed but the output
        # will be discarded through the hook. It is useful for collecting activation stats,
        # for example during activation calibration in PTQ
        for module in model.modules():
            if isinstance(module, _ACC_PROXIES):
                module.train(is_training)
                previous_state[module] = module.disable_quant
                module.disable_quant = disable_quant
            elif isinstance(module, ActQuantProxyFromInjectorBase):
                module.train(is_training)
                previous_state[module] = module.disable_quant
                for m in module.modules():
                    if hasattr(m, 'observer_only'):
                        previous_observer_state[m] = m.observer_only
                        m.observer_only = disable_quant and call_act_quantizer_impl
                # When call_act_quantizer_impl=False, quantization is not disabled
                module.disable_quant = disable_quant and not call_act_quantizer_impl
        return previous_state, previous_observer_state

    @staticmethod
    def _set_param_quantization(
        model: nn.Module,
        disable_quant: bool,
        is_training: bool,
        quant_proxies: Tuple[Type[ParameterQuantProxyFromInjector]] = _PARAM_PROXIES
    ) -> Dict[nn.Module, bool]:
        # Save previous state of parameter quantizers
        previous_state = {}
        for module in model.modules():
            if isinstance(module, quant_proxies):
                module.train(is_training)
                previous_state[module] = module.disable_quant
                module.disable_quant = disable_quant
        return previous_state

    @staticmethod
    def _restore_quantization_state(
            model: nn.Module,
            previous_state: Dict[nn.Module, bool],
            is_training: bool,
            quant_proxies: Tuple[Type[QuantProxyFromInjector]]) -> None:
        for module in model.modules():
            if isinstance(module, quant_proxies) and module in previous_state:
                module.train(is_training)
                module.disable_quant = previous_state[module]

    @staticmethod
    def disable_return_quant_tensor(model: nn.Module) -> Dict[nn.Module, bool]:
        previous_state = {}
        for module in model.modules():
            if hasattr(module, 'return_quant_tensor'):
                previous_state[module] = module.return_quant_tensor
                module.return_quant_tensor = False
        return previous_state

    @staticmethod
    def restore_return_quant_tensor(
            model: nn.Module, previous_state: Dict[nn.Module, bool]) -> None:
        for module in model.modules():
            if hasattr(module, 'return_quant_tensor') and module in previous_state:
                module.return_quant_tensor = previous_state[module]

    @staticmethod
    def disable_act_quant_hook(
            module: nn.Module, inp: Union[tuple, QuantTensor],
            output: torch.Tensor) -> torch.Tensor:
        inp = unpack_input(inp)
        if module.fused_activation_quant_proxy is not None:
            inp = module.fused_activation_quant_proxy.activation_impl(inp)
        # consider the first module as representative for the activation fn
        # as this is what would happen with a shared act_quant
        # this gets called both during (empty) input_quant and act_quant
        # but for HardTanh it's not an issue
        if isinstance(module.tracked_module_list[0], QuantHardTanh):
            inp = F.hardtanh(
                inp, min_val=module.quant_injector.min_val, max_val=module.quant_injector.max_val)
        return inp

    @staticmethod
    def disable_act_quantization(
            model: nn.Module,
            is_training: bool,
            call_act_quantizer_impl: bool = False) -> Dict[nn.Module, bool]:
        return QuantizationStatusManager._set_act_quantization(
            model=model,
            disable_quant=True,
            is_training=is_training,
            call_act_quantizer_impl=call_act_quantizer_impl,
        )

    @staticmethod
    def enable_act_quantization(model: nn.Module, is_training: bool) -> None:
        QuantizationStatusManager._set_act_quantization(
            model=model,
            disable_quant=False,
            is_training=is_training,
        )

    @staticmethod
    def restore_act_quantization(
            model: nn.Module,
            previous_observer_state: Dict[nn.Module, bool],
            previous_state: Dict[nn.Module, bool],
            is_training: bool) -> None:
        QuantizationStatusManager._restore_quantization_state(
            model=model,
            previous_state=previous_state,
            is_training=is_training,
            quant_proxies=_ACC_PROXIES + (ActQuantProxyFromInjectorBase,),
        )
        for module in model.modules():
            if isinstance(module, ActQuantProxyFromInjectorBase) and module in previous_state:
                for m in module.modules():
                    if hasattr(m, 'observer_only'):
                        m.observer_only = previous_observer_state[m]

    @staticmethod
    def disable_param_quantization(model: nn.Module, is_training: bool) -> Dict[nn.Module, bool]:
        return QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=True,
            is_training=is_training,
            quant_proxies=_PARAM_PROXIES,
        )

    @staticmethod
    def enable_param_quantization(model: nn.Module, is_training: bool) -> None:
        QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=False,
            is_training=is_training,
            quant_proxies=_PARAM_PROXIES,
        )

    @staticmethod
    def restore_param_quantization(
            model: nn.Module, previous_state: Dict[nn.Module, bool], is_training: bool) -> None:
        QuantizationStatusManager._restore_quantization_state(
            model=model,
            previous_state=previous_state,
            is_training=is_training,
            quant_proxies=_PARAM_PROXIES,
        )

    @staticmethod
    def disable_weight_quantization(model: nn.Module, is_training: bool) -> Dict[nn.Module, bool]:
        return QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=True,
            is_training=is_training,
            quant_proxies=_WEIGHT_PROXIES,
        )

    @staticmethod
    def enable_weight_quantization(model: nn.Module, is_training: bool) -> None:
        QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=False,
            is_training=is_training,
            quant_proxies=_WEIGHT_PROXIES,
        )

    @staticmethod
    def restore_weight_quantization(
            model: nn.Module, previous_state: Dict[nn.Module, bool], is_training: bool) -> None:
        QuantizationStatusManager._restore_quantization_state(
            model=model,
            previous_state=previous_state,
            is_training=is_training,
            quant_proxies=_WEIGHT_PROXIES,
        )

    @staticmethod
    def disable_bias_quantization(model: nn.Module, is_training: bool) -> Dict[nn.Module, bool]:
        return QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=True,
            is_training=is_training,
            quant_proxies=_BIAS_PROXIES,
        )

    @staticmethod
    def enable_bias_quantization(model: nn.Module, is_training: bool) -> None:
        QuantizationStatusManager._set_param_quantization(
            model=model,
            disable_quant=False,
            is_training=is_training,
            quant_proxies=_BIAS_PROXIES,
        )

    @staticmethod
    def restore_bias_quantization(
            model: nn.Module, previous_state: Dict[nn.Module, bool], is_training: bool) -> None:
        QuantizationStatusManager._restore_quantization_state(
            model=model,
            previous_state=previous_state,
            is_training=is_training,
            quant_proxies=_BIAS_PROXIES,
        )


class quantization_status_manager:
    """
        Context manager to disable/enable quantization for a nn.Module,
        potentially excluding a set of submodules.

    Args:
        model (nn.Module): module for which quantization will be enabled/
            disabled
        is_training (bool): whether to set the module in training mode on
            __enter__
        call_act_quantizer_impl (bool): if set to True, activation quantization
            is performed, but the output is discarded. Useful for collecting
            activation stats, for example during activation calibration in PTQ
        disable_quant_act (bool): whether to disable activation quantization
        disable_weight_quant (bool): whether to disable weight quantization
        disable_bias_quant (bool): whether to disable bias quantization
    """

    def __init__(
            self,
            model: nn.Module,
            call_act_quantizer_impl: bool = False,
            disable_act_quant: bool = False,
            disable_weight_quant: bool = False,
            disable_bias_quant: bool = False,
            is_training: Optional[bool] = None):
        self.model = model
        self.call_act_quantizer_impl = call_act_quantizer_impl
        # Flags to disable quantization in a fined-grained manner
        # Activations
        self.disable_act_quant = disable_act_quant
        self.act_quant_state = {}
        self.act_quant_observer_state = {}
        # Weights
        self.disable_weight_quant = disable_weight_quant
        self.weight_quant_state = {}
        # Bias
        self.disable_bias_quant = disable_bias_quant
        self.bias_quant_state = {}
        # Return QuantTensor
        self.disable_return_quant_tensor = disable_act_quant or disable_weight_quant
        self.return_quant_tensor_state = {}
        # If is_training is not specified, the value of model.training is used
        self.is_training = is_training if is_training is not None else model.training
        self.prev_is_training_state = model.training

    def disable_module_quantization(self, module: nn.Module) -> None:
        if self.disable_act_quant:
            self.act_quant_state, self.act_quant_observer_state = QuantizationStatusManager.disable_act_quantization(
                model=module,
                is_training=self.is_training,
                call_act_quantizer_impl=self.call_act_quantizer_impl,
            )
        if self.disable_weight_quant:
            self.weight_quant_state = QuantizationStatusManager.disable_weight_quantization(
                model=module,
                is_training=self.is_training,
            )
        if self.disable_bias_quant:
            self.bias_quant_state = QuantizationStatusManager.disable_bias_quantization(
                model=module,
                is_training=self.is_training,
            )
        if self.disable_return_quant_tensor:
            self.return_quant_tensor_state = QuantizationStatusManager.disable_return_quant_tensor(
                module)

    def enable_module_quantization(self, module: nn.Module) -> None:
        if self.disable_act_quant:
            QuantizationStatusManager.restore_act_quantization(
                model=module,
                previous_state=self.act_quant_state,
                previous_observer_state=self.act_quant_observer_state,
                is_training=self.prev_is_training_state,
            )
        if self.disable_weight_quant:
            QuantizationStatusManager.restore_weight_quantization(
                model=module,
                previous_state=self.weight_quant_state,
                is_training=self.prev_is_training_state,
            )
        if self.disable_bias_quant:
            QuantizationStatusManager.restore_bias_quantization(
                model=module,
                previous_state=self.bias_quant_state,
                is_training=self.prev_is_training_state,
            )
        if self.disable_return_quant_tensor:
            QuantizationStatusManager.restore_return_quant_tensor(
                module, self.return_quant_tensor_state)

    def __enter__(self):
        self.disable_module_quantization(module=self.model)

    def __exit__(self, type, value, traceback):
        self.enable_module_quantization(module=self.model)


class calibration_mode(quantization_status_manager):

    def __init__(self, model, enabled=True):
        super().__init__(
            model=model,
            call_act_quantizer_impl=True,
            disable_act_quant=True,
            disable_weight_quant=True,
            disable_bias_quant=True,
            is_training=True)
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            # Call __enter__ on quantization_status_manager context manager
            super().__enter__()
            self.model.apply(extend_collect_stats_steps)
            self.model.apply(set_collect_stats_to_average)

    def __exit__(self, type, value, traceback):
        # Call __exit__ on quantization_status_manager context manager
        super().__exit__(type, value, traceback)
        self.model.apply(finalize_collect_stats)


class _BiasCorrection:

    LAYERS = (QuantWBIOL,)

    def __init__(self, layers=LAYERS, skip_if_no_bias=False):
        self.layers = layers
        self.iterations = {}
        self.correction_map = {}
        self.float_mean_map = {}
        self.collect_float_mean_hooks = []
        self.correct_bias_hooks = []
        self.skip_if_no_bias = skip_if_no_bias

    def compute_mean(self, inp, transpose_dim):
        inp = inp.transpose(0, transpose_dim)
        return inp.reshape(inp.shape[0], -1).mean(dim=1).detach()

    def channel_dim(self, inp, module):
        if len(inp.shape) == 3 and isinstance(module, QuantLinear):
            channel_dim = 2
        else:
            channel_dim = 1
        return channel_dim

    def collect_float_mean(self, module, inp, name):
        inp = unpack_input(inp)
        if name in self.float_mean_map.keys():
            raise RuntimeError("Module to bias-correct called multiple times, not supported.")
        transpose_dim = self.channel_dim(inp, module)
        self.float_mean_map[name] = self.compute_mean(inp, transpose_dim)

    def update_correction(self, name, error):
        if name not in self.correction_map:
            self.correction_map[name] = error
        else:
            self.correction_map[name] += error

    def apply_correction(self, model):
        for name, module in model.named_modules():
            if name in self.correction_map.keys():
                correction = self.correction_map[name] / self.iterations[name]
                # When accelerate is enabled, bring tensors onto the device to avoid allocating a meta parameter.
                if hasattr(module, 'allocate_params'):
                    module.allocate_params(module)
                if module.bias is not None:
                    module.bias.data += correction
                elif self.skip_if_no_bias is False:
                    # If accelerate is enabled, bias will be on the same execution device as the weights, but won't be managed properly by accelerate
                    module.register_parameter(
                        'bias', nn.Parameter(correction).to(module.weight.device))
                # Offload params again
                if hasattr(module, 'offload_params'):
                    module.offload_params(module)

    def compute_correct_bias(self, module, inp, name):
        inp = unpack_input(inp)
        if name in self.float_mean_map.keys():
            transpose_dim = self.channel_dim(inp, module)
            quant_mean = self.compute_mean(inp, transpose_dim)
            error = self.float_mean_map[name] - quant_mean
            self.update_correction(name, error)
            del self.float_mean_map[name]

    def register_hook_to_wbiol(self, model, hooks):
        """
        Forward hooks are registered to the WBIOL layers.
        In this way, if more hooks are registered to these layers, they will be only called once per module call.
        We perform two more forwards for each WBIOL, but since we call forward and not __call__, eventual hooks would be skipped.
        This is a desired behaviour, since these extra forwards are only necessary to compute the bias correction factor.

        If we registered a single hook for the entire model, and hooks were added to WBIOL layers,
        these would be called by our extra forwards, which would be an unexpected behaviours.
        """
        for name, module in model.named_modules():
            if isinstance(module, self.layers):
                self.iterations[name] = 0
                hook_fn = partial(self.forward_hook_wbiol, name=name)
                hooks.append(module.register_forward_hook(hook_fn))

    def forward_hook_wbiol(self, module, inp, output, name):
        """
        After each forward, we perform two extra forwards to register float and quant outputs
        and the error between the two.

        We do not return the original quant output, but the float one, to avoid error accumulation
        """
        # Compute float reference
        # TODO: Ensure that removing is_training=False does not cause any regression and remove,
        # if that is the case
        with quantization_status_manager(
                module,
                disable_act_quant=True,
                disable_weight_quant=True,
                disable_bias_quant=True,
                is_training=False,
        ):
            out_float = module.forward(*inp)  # Required to avoid infinite recursion
        self.collect_float_mean(module, out_float, name)
        # Keep output quant disabled until further notice
        QuantizationStatusManager.disable_act_quantization(
            model=module.output_quant, is_training=False)
        out_quant = output
        self.compute_correct_bias(module, out_quant, name)
        self.iterations[name] += 1
        return out_float


class norm_correction_mode:

    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            for m in self.model.modules():
                if isinstance(m, BN_LAYERS):
                    m.register_buffer("old_running_mean", m.running_mean.clone())
                    m.register_buffer("old_running_var", m.running_var.clone())
                    m.reset_running_stats()
                    m.momentum = None
                    m.train()

    def __exit__(self, type, value, traceback):
        if self.enabled:
            for m in self.model.modules():
                if isinstance(m, BN_LAYERS):
                    a = torch.sqrt(m.old_running_var / m.running_var)
                    b = m.old_running_mean - a * m.running_mean
                    m.running_var = m.old_running_var.clone() / a
                    m.running_mean = (m.old_running_mean.clone() - b) / a
                    del m.old_running_var
                    del m.old_running_mean
