import logging
from typing import Mapping

from accelerate.hooks import add_hook_to_module
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import check_tied_parameters_in_config
from accelerate.utils import compute_module_sizes
from accelerate.utils import find_device
from accelerate.utils import find_tied_parameters
from accelerate.utils import get_max_layer_size
from accelerate.utils import get_max_memory
from accelerate.utils import send_to_device
from accelerate.utils.modeling import named_module_tensors
import torch

from brevitas.graph.utils import get_module

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


# The main rationale is to work around the fact that module.__class__.__name__ is Module for everything
# We do not need to keep entire blocks together anymore, since we add a functional equivalent of the AlignDeviceHook
# before every call function.
# This should also work across multiple GPUs (not battle-tested).
def infer_fx_auto_device_map(
    model,
    max_memory=None,
    dtype=None,
    special_dtypes=None,
    verbose=False,
):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
    """
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
    all_modules = [n.target for n in list(model.graph.nodes) if n.op == 'call_module']
    for node in model.graph.nodes:
        # If it's a module, we simply offload it or move it to the desired device
        if node.op == 'call_module':
            name = node.target
            module = get_module(model, node.target)
            call_list.append((name, module))
        # If it's get_attr, we check what module it is attached to
        # In case the module is not part of call_module, we specifically allocate the buffer/parameter on some device
        # NB: This does NOT guarantee that it will be aligned with whatever input tensor it will be combined with
        # For that, there is a separate function
        if node.op == 'get_attr':
            target = node.target
            if target in buffers_attributes:
                module_name = '.'.join(target.split('.')[:-1])
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
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                module_sizes, [])
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        tied_param_goups = [
            tied_group for tied_group in tied_parameters
            if any(name + '.' in k + '.'
                   for k in tied_group) and not all(name + '.' in k + '.' for k in tied_group)]

        if verbose and len(tied_param_goups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_goups}")
        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + '.' not in p + '.'] for tied_group in tied_param_goups
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
        device_map = {'': list(device_map.values())[0]}
    return device_map


def offload_call_function(model, device_map):

    # If we only have one device, offloading is not needed
    if len(set(device_map.values())) == 1:
        return

    for node in model.graph.nodes:
        if node.op == 'call_function':

            def new_func(*args, old_callable=node.target, **kwargs):
                args = list(args)
                device_mapping = dict()

                # Identify the device for each tensor in args and kwargs
                for i, arg in enumerate(args):
                    all_devices = find_all_devices(arg)
                    if all_devices is not None:
                        device_mapping.update(dict(all_devices))

                for k, v in kwargs.items():
                    all_devices = find_all_devices(arg)
                    if all_devices is not None:
                        device_mapping.update(dict(all_devices))

                total_devices = [d for d in list(device_mapping.values()) if d is not None]
                # If there is only one device, no re-alignement is necessary
                if len(set(total_devices)) > 1:
                    # Pick the main device, i.e. the first device that is not 'cpu' or 'disk'
                    if set(device_mapping.values()) == {"cpu"} or set(device_mapping.values()) == {
                            "cpu", "disk"}:
                        device = "cpu"
                    else:
                        device = [d for d in device_mapping.values() if d not in ["cpu", "disk"]][0]
                    # Align args and kwargs to the same device
                    args = send_to_device(args, device)
                    kwargs = send_to_device(kwargs, device)
                out = old_callable(*args, **kwargs)
                if len(set(total_devices)) > 1:
                    # Restore the original device to avoid memory leaks
                    for k, v in device_mapping.items():
                        k = k.to(v)

                return out

            node.meta['orig_target'] = node.target
            node.target = new_func

    model.recompile()
    model.graph.lint()


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
