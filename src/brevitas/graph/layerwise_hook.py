# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from operator import attrgetter
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import warnings

from torch.fx import GraphModule as TorchGraphModule

from brevitas.fx import GraphModule
from brevitas.graph.calibrate import quantization_status_manager


@dataclass
class LayerHandler:
    layer_names: Set = field(default_factory=set)
    forward_count: int = 0


class layerwise_hook_mode(quantization_status_manager):
    """Base context manager for layer-wise calibration driven by forward hooks.

    Shared by GPxQ (GPTQ / GPFQ / MagR / Qronos) and PiSO scale optimization
    (optimize_scale_mode). On entry it selects the supported layers, attaches a
    per-layer forward-pre-hook that accumulates statistics as calibration data
    flows through, and (optionally) swaps in a forward that catches the early-stop
    exception. Once enough data has been fed, update() runs the per-layer solver
    and removes the hooks.

    Layers can be grouped for parallel processing via group_of_parallel_layers:
    each group shares a single optimizer and its members' hooks all reference the
    same LayerHandler.

    Subclasses implement _is_module_supported, initialize_module_optimizer and
    catch_stopfwd; the no-op _pre_enter / _post_enter / _post_exit /
    _after_layer_update extension points let them run extra bookkeeping without
    duplicating the lifecycle.
    """

    def __init__(
            self,
            model,
            disable_act_quant: bool,
            disable_bias_quant: bool,
            create_weight_orig: bool = False,
            group_of_parallel_layers: Optional[List[str]] = None,
            swap_forward: bool = True) -> None:
        super().__init__(
            model=model,
            disable_act_quant=disable_act_quant,
            disable_weight_quant=False,
            disable_bias_quant=disable_bias_quant,
        )
        self.create_weight_orig = create_weight_orig
        self.group_of_parallel_layers = group_of_parallel_layers

        self.hook_dict = dict()
        # per-group optimizer, keyed by group name
        self.layers = dict()
        # reference shared by all hooks to track which layers received data
        self.current_layer = LayerHandler()
        self.num_layers = 0

        # Save the original forward so it can be restored on exit; optionally swap
        # in catch_stopfwd to intercept the early-stop exception.
        self.orig_forward = self.model.forward
        self._forward_swapped = swap_forward
        if swap_forward:
            self._set_forward(self.catch_stopfwd)

    def _set_forward(self, forward_fn) -> None:
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            self.model.__class__.forward = forward_fn
        else:
            self.model.forward = forward_fn

    def _collect_layer_groups(self) -> Dict[str, List[Tuple[str, object]]]:
        """Map each group name to its list of (name, module) members.

        By default every supported module is its own group. When
        group_of_parallel_layers is set, the named layers are folded into a single
        group that shares one optimizer.
        """
        dict_of_layers = {
            name: [(name, module)] for name,
            module in self.model.named_modules() if self._is_module_supported(module)}
        if self.group_of_parallel_layers is not None:
            for parallel_layers in self.group_of_parallel_layers:
                for name in parallel_layers:
                    if name not in dict_of_layers:
                        raise ValueError(
                            "The layer {} is not present in the model or it is not supported"
                            .format(name))
                    del dict_of_layers[name]
                names = '_'.join(parallel_layers)
                dict_of_layers[names] = [
                    (name, attrgetter(name)(self.model)) for name in parallel_layers]
        return dict_of_layers

    def _warn_on_existing_hooks(self, dict_of_layers: Dict[str, List[Tuple[str, object]]]) -> None:
        # The normal forward flow is highly disrupted during calibration, so warn
        # if any module already carries hooks.
        for _, parallel_layers in dict_of_layers.items():
            for name, module in parallel_layers:
                if len(module._forward_hooks) > 0 or len(module._forward_pre_hooks):
                    warnings.warn(
                        f'Hooks detected during setup. '
                        f'Behaviour might deviate from what expected.')

    def __enter__(self):
        # Disable quantization selectively
        super().__enter__()
        self._pre_enter()

        dict_of_layers = self._collect_layer_groups()
        self._warn_on_existing_hooks(dict_of_layers)

        # One optimizer per member module (keyed by its own name), sharing the
        # group's len_parallel_layers and the same LayerHandler.
        for _, parallel_layers in dict_of_layers.items():
            for name, module in parallel_layers:
                module_optimizer = self.initialize_module_optimizer(
                    module,
                    name,
                    len_parallel_layers=len(parallel_layers),
                    create_weight_orig=self.create_weight_orig)
                hook_fn = partial(module_optimizer.update_batch, current_layer=self.current_layer)
                self.hook_dict[name] = module.register_forward_pre_hook(hook_fn)
                self.layers[name] = module_optimizer

        self._post_enter(dict_of_layers)
        self.num_layers = len(dict_of_layers)
        return self

    def __exit__(self, type, value, traceback):
        # Restore original quantization configuration
        super().__exit__(type, value, traceback)
        if self._forward_swapped:
            self._set_forward(self.orig_forward)
        self._post_exit()

    def update(self):
        for name in self.current_layer.layer_names:
            self.layers[name].single_layer_update()
            self.hook_dict[name].remove()
            self._after_layer_update(name)
        self.current_layer.layer_names.clear()

    # -------------------------------------------------------------------------
    # Extension points (no-op defaults).
    # -------------------------------------------------------------------------

    def _pre_enter(self) -> None:
        pass

    def _post_enter(self, dict_of_layers: Dict[str, List[Tuple[str, object]]]) -> None:
        pass

    def _post_exit(self) -> None:
        pass

    def _after_layer_update(self, name: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # Abstract hooks implemented by subclasses.
    # -------------------------------------------------------------------------

    @abstractmethod
    def _is_module_supported(self, module) -> bool:
        pass

    @abstractmethod
    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        pass

    @abstractmethod
    def catch_stopfwd(self, *args, **kwargs):
        pass
