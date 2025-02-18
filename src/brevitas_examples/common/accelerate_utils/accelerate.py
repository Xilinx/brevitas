# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Dict, Mapping, Optional, Union

from accelerate import dispatch_model
from accelerate import infer_auto_device_map
from accelerate.hooks import add_hook_to_module
from accelerate.hooks import AlignDevicesHook
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import check_tied_parameters_in_config
from accelerate.utils import compute_module_sizes
from accelerate.utils import find_tied_parameters
from accelerate.utils import get_max_layer_size
from accelerate.utils import get_max_memory
from accelerate.utils import send_to_device
from accelerate.utils.modeling import named_module_tensors
from psutil import virtual_memory
import torch

import brevitas.config as config
from brevitas.graph.utils import get_module
from brevitas.utils.python_utils import recurse_getattr

logger = logging.getLogger(__name__)


def align_input(model, device_map):
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]
    hook = AlignDevicesHook(
        execution_device=main_device, io_same_device=True, skip_keys=None, tied_params_map=None)
    add_hook_to_module(model, hook)
    return model


# Adapted from accelerate.utils.modeling.infer_auto_device_map
# Licensed under Apache License 2.0.
def infer_fx_auto_device_map(
    model: torch.fx.GraphModule,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.dtype]]] = None,
    verbose: bool = False,
):
    """
    Extends accelerate's infer_auto_device_map function to be compatible with torch.fx.GraphModule.

    The main modifications are:
    - Work around the fact that module.__class__.__name__ is Module for everything
    - We do not need to keep entire blocks together anymore, since we add a functional equivalent of the AlignDeviceHook
    before every call function.
    """
    # TODO: Why no no_split_module_classes, clean_result parameters?

    # Get default / clean up max_memory
    max_memory = get_max_memory(max_memory)

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")
    gpus = [device for device in devices if device not in ["cpu", "disk"]]

    # Devices that need to keep space for a potential offloaded layer.
    if "mps" in gpus:
        main_devices = ["mps"]
    elif len(gpus) > 0:
        main_devices = [gpus[0], "cpu"]
    else:
        main_devices = ["cpu"]

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_parameters) == 0:
        logger.warn(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )

    device_map = {}
    current_device = 0
    current_memory_used = 0

    call_list = []
    buffers_attributes = [n for n, _ in list(named_module_tensors(model, recurse=True))]
    all_modules = [n.target for n in list(model.graph.nodes) if n.op == "call_module"]
    for node in model.graph.nodes:
        # If it's a module, we simply offload it or move it to the desired device
        if node.op == "call_module":
            name = node.target
            module = get_module(model, node.target)
            call_list.append((name, module))
        # If it's get_attr, we check what module it is attached to
        # In case the module is not part of call_module, we specifically allocate the buffer/parameter on some device
        # NB: This does NOT guarantee that it will be aligned with whatever input tensor it will be combined with
        # For that, there is a separate function
        if node.op == "get_attr":
            target = node.target
            if target in buffers_attributes:
                module_name = ".".join(target.split(".")[:-1])
                if module_name not in all_modules:
                    module = get_module(model, target)
                    call_list.append((target, module))

    # Direct submodules and parameters
    modules_to_treat = call_list

    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, [])

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if n != name and not n.startswith(name + ".")]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)], module_sizes, []
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        tied_param_goups = [
            tied_group for tied_group in tied_parameters
            if any(name + "." in k + "."
                   for k in tied_group) and not all(name + "." in k + "." for k in tied_group)]

        if verbose and len(tied_param_goups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_goups}")
        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_goups
            ], [])
        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
        # Case 1 -> We're too big!
        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            # For FX, we never split a leaf call_module
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size-current_memory_used}, module size {module_size}).")

            if verbose:
                print("This module cannot be split, going to the next device.")
            current_device += 1
            modules_to_treat = [(name, module)] + modules_to_treat
            current_memory_used = 0

        # Case 2, it fits! We're not entirely out of the wood though, because we may have some tied parameters.
        elif len(tied_params) > 0:
            # First locate all tied modules
            tied_module_names = []
            tied_modules = []
            for tied_param in tied_params:
                tied_module_index = [
                    i for i, (n, _) in enumerate(modules_to_treat) if n in tied_param][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])
            if verbose:
                print(
                    f"  It looks like {name} is going to fit on {devices[current_device]} but we have tied "
                    f"parameters to account for.\n  - Names {tied_params}\n  - Module names {tied_module_names}"
                )

            # Let's see if it all fits first
            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

            if current_max_size is None or current_memory_used + module_size_with_ties <= current_max_size:
                # We really really fit!
                if verbose:
                    print(f"Putting {name} and {tied_module_names} on {devices[current_device]}.")
                current_memory_used += module_size_with_ties
                device_map[name] = devices[current_device]
                for tied_module_name in tied_module_names:
                    if tied_module_name in [m[0] for m in modules_to_treat]:
                        # The module may have been removed by a previous iteration of this loop.
                        tied_module_index = [
                            i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name
                        ][0]
                        modules_to_treat.pop(tied_module_index)
                    device_map[tied_module_name] = devices[current_device]

            else:
                # We don't fit with the tied modules. Next question is: can we split one of the tied modules to make it
                # smaller or do we need to go on the next device?
                if verbose:
                    print(
                        f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                        f"available {current_max_size-current_memory_used}, needed size {module_size_with_ties})."
                    )
                split_happened = False
                for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                    tied_module_children = list(tied_module.named_children())
                    if len(tied_module_children) == 0:
                        # can't break this one.
                        continue
                    if verbose:
                        print(f"Splitting {tied_module_name}.")
                    tied_module_children = list(
                        tied_module.named_parameters(recurse=False)) + tied_module_children
                    tied_module_children = [
                        (f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                    tied_module_index = [
                        i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                    modules_to_treat = ([(name, module)] + modules_to_treat[:tied_module_index] +
                                        tied_module_children +
                                        modules_to_treat[tied_module_index + 1:])
                    # Update the max layer size.
                    max_layer_size, max_layer_names = get_max_layer_size(
                        [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                        module_sizes,
                        [],
                    )
                    split_happened = True
                    break

                if not split_happened:
                    # If the tied module is not split, we go to the next device
                    if verbose:
                        print("None of the tied module can be split, going to the next device.")
                    current_device += 1
                    modules_to_treat = [(name, module)] + modules_to_treat
                    current_memory_used = 0

        else:
            if verbose:
                if current_max_size is None:
                    print(f"Putting {name} (size={module_size}) on {devices[current_device]}.")
                else:
                    print(
                        f"Putting {name} (size={module_size}) on {devices[current_device]} "
                        f"(available={current_max_size-current_memory_used}).")
            current_memory_used += module_size
            device_map[name] = devices[current_device]

    # If we have only one device, we simplify the device_map
    if len(set(device_map.values())) == 1:
        device_map = {"": list(device_map.values())[0]}
    return device_map


def offload_call_function(model: torch.fx.GraphModule, device_map: Dict):
    """
    Attaches AlignDevicesHook to fx.GraphModule call_function nodes. Although accelerate's `offload_model` attaches hooks
    to submodules, it is unable to detect call_function.
    """
    # If we only have one device, offloading is not needed
    # if len(set(device_map.values())) == 1:
    #     return

    for node in model.graph.nodes:

        if node.op == "call_function":

            def new_func(*args, old_callable=node.target, **kwargs):
                args = list(args)
                device_mapping = {}

                # Identify the device for each tensor in args and kwargs
                for _, arg in enumerate(args):
                    all_devices = find_all_devices(arg)
                    if all_devices is not None:
                        device_mapping.update(dict(all_devices))

                for k, v in kwargs.items():
                    all_devices = find_all_devices(k)
                    if all_devices is not None:
                        device_mapping.update(dict(all_devices))

                total_devices = [d for d in list(device_map.values()) if d is not None]

                # Pick the main device, i.e. the first device that is not 'cpu' or 'disk'
                if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu",
                                                                                       "disk"}:
                    device = "cpu"
                else:
                    device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]
                # Align args and kwargs to the same device
                args = send_to_device(args, device)
                kwargs = send_to_device(kwargs, device)

                out = old_callable(*args, **kwargs)

                if len(set(total_devices)) > 1:
                    # Restore the original device to avoid memory leaks
                    for k, v in device_mapping.items():
                        k = k.to(v)

                return out

            node.meta["orig_target"] = node.target
            node.target = new_func

    model.recompile()
    model.graph.lint()


def remove_hooks(model: torch.nn.Module):
    for module in model.modules():
        if hasattr(module, "_hf_hook"):
            if hasattr(module, "allocate_params"):
                del module.allocate_params
            if hasattr(module, "offload_params"):
                del module.offload_params
    remove_hook_from_module(model, recurse=True)
    model.cpu()
    if hasattr(model, "graph"):
        for node in model.graph.nodes:
            if node.op == "call_function":
                if "orig_target" in node.meta:
                    node.target = node.meta["orig_target"]
                    del node.meta["orig_target"]
        model.recompile()
        model.graph.lint()


def update_internal_dict(module, *args, **kwargs):
    prefix = module._hf_hook.weights_map.prefix
    for key in module.state_dict().keys():
        # It might happen that we call an quantization's inner modules, and this cause some parameters to be
        # already on meta device. This is not a problem for their value but we need to check here
        curr_device = (recurse_getattr(module, key + ".data")).device
        if str(curr_device) != "meta":
            module._hf_hook.weights_map.dataset.state_dict[prefix + key] = (
                recurse_getattr(module, key + ".data")).cpu()


def find_all_devices(data):
    """
    Finds the device on which a nested dict/list/tuple of tensors lies (assuming they are all on the same device).
    Args:
        (nested list/tuple/dictionary of `torch.Tensor`): The data we want to know the device of.
    """
    if isinstance(data, Mapping):
        devices = []
        for obj in data.values():
            device = find_all_devices(obj)
            if device is not None:
                devices.extend(device)
        return devices
    elif isinstance(data, (tuple, list)):
        devices = []
        for obj in data:
            device = find_all_devices(obj)
            if device is not None:
                devices.extend(device)
        return devices
    elif isinstance(data, torch.Tensor):
        return [(data, str(data.device))]


def calc_gpu_device_map(absolute_mem_margin: float = 2.0 * 1e9,
                        relative_mem_margin: float = 0.3) -> Dict[int, float]:
    torch.cuda.empty_cache()
    gpu_device_map = {
        i: (torch.cuda.mem_get_info(i)[0] - absolute_mem_margin) * (1.0 - relative_mem_margin)
        for i in range(torch.cuda.device_count())}
    return gpu_device_map


def calc_cpu_device_map(absolute_mem_margin: float = 2.0 * 1e9,
                        relative_mem_margin: float = 0.3) -> Dict[str, float]:
    cpu_device_map = {
        "cpu": (virtual_memory().available - absolute_mem_margin) * (1.0 - relative_mem_margin)}
    return cpu_device_map


def offload_model(
    model: torch.nn.Module,
    gpu_device_map: Optional[Dict[int, float]] = None,
    cpu_device_map: Optional[Dict[str, float]] = None,
) -> torch.nn.Module:
    """
    Wraps accelerate's infer_auto_device_map and dispatch_model.

    This functions if compatible both with classic nn.Modules, and with torch.fx.GraphModule.
    """

    # FX vs non-FX model need different offloading
    config._FULL_STATE_DICT = True
    if gpu_device_map is None:
        gpu_device_map = calc_gpu_device_map()
    if cpu_device_map is None:
        cpu_device_map = calc_cpu_device_map()
    memory_map = {**cpu_device_map, **gpu_device_map}

    if isinstance(model, torch.fx.GraphModule):
        device_map = infer_fx_auto_device_map(model, memory_map)
        offload_call_function(model, device_map)
    else:
        # Some models do no have the attribute _no_split_modules, so a check is needed to prevent
        # this call to crash.
        device_map = infer_auto_device_map(
            model,
            memory_map,
            no_split_module_classes=model._no_split_modules
            if hasattr(model, "_no_split_modules") else None)

    model = dispatch_model(model, device_map)

    # Fixes an asymetric behavior in Accelerate where hooks are not attached at all when a single device is used.
    # TODO: Fix directly in accelerate.
    if len(set(device_map.values())) == 1:
        model = align_input(model, device_map)

    config._FULL_STATE_DICT = False
    if "disk" in model.hf_device_map.values():
        raise ValueError("Disk offload is not supported with quantization.")

    # We attach these functions to the hooked modules for convenience when modifying parameters during PTQ (e.g. SmoothQuant).
    # Attaching these functions allows use to fix a bug in accelerate with offloading to RAM/disk where even though a submodule parameter is updated, it is actually not updated in the AlignDevicesHook `weights_map` and thus
    # the update is ignored elsewhere.
    # TODO: Fix this bug directly in accelerate. https://github.com/huggingface/accelerate/pull/2214 would fix the bug for RAM offliading.
    def allocate_params(module):
        """
        This function calls the pre_forward function of the _hf_hook, making sure parameters are on
        the selected device, rather than on the meta device.
        """
        if module._hf_hook.offload is False:
            return
        # When quantizing and retrieving parameters (e.g., during GPTQ), we want to recurse through
        # all the submodules
        for m in module.modules():
            if hasattr(m, "_hf_hook"):
                m._hf_hook.pre_forward(m)

    def offload_params(module):
        """
        This functions moves the parameters back to the meta device, after making sure to update the
        internal state dict with the most recent values.
        """
        if module._hf_hook.offload is False:
            return
        update_internal_dict(module)
        for m in module.modules():
            if hasattr(m, "_hf_hook"):
                m._hf_hook.post_forward(m, torch.tensor([]))

    for module in model.modules():
        if hasattr(module, "_hf_hook"):
            module.allocate_params = allocate_params
            module.offload_params = offload_params

    return model
