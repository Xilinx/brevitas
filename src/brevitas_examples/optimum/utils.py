from accelerate import dispatch_model
from accelerate import infer_auto_device_map
from accelerate.hooks import remove_hook_from_module
from psutil import virtual_memory
import torch

import brevitas.config as config
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.optimum.offloading_utils import infer_fx_auto_device_map
from brevitas_examples.optimum.offloading_utils import offload_call_function


def maybe_offload_weights_to_cpu(model, is_fx=False):
    config._FULL_STATE_DICT = True
    cuda_device_map = {
        i: torch.cuda.mem_get_info(i)[0] * 0.7 for i in range(torch.cuda.device_count())}
    cpu_device_map = {'cpu': virtual_memory().available * 0.7}
    memory_map = {**cpu_device_map, **cuda_device_map}
    if is_fx:
        device_map = infer_fx_auto_device_map(model, memory_map)
        device_map = offload_call_function(model, memory_map, device_map)
    else:
        device_map = infer_auto_device_map(
            model, memory_map, no_split_module_classes=model._no_split_modules)
    model = dispatch_model(model, device_map)
    config._FULL_STATE_DICT = False
    if "disk" in model.hf_device_map.values():
        raise ValueError("disk offload is not supported with quantization")

    ### Not sure about the benefits of this part

    # if "cpu" in model.hf_device_map.values():
    #     hook = None
    #     for name, device in model.hf_device_map.items():
    #         if device == 'cpu':
    #             module = recurse_getattr(model, name)
    #             remove_hook_from_module(module, recurse=True)
    #             module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
    return model


def remove_hooks(model):
    for module in model.modules():
        if hasattr(module, '_hf_hook'):
            del module.allocate_params
            del module.offload_params
    remove_hook_from_module(model, recurse=True)
    model.cpu()
    if hasattr(model, 'graph'):
        for node in model.graph.nodes:
            if node.op == 'call_function':
                if 'orig_target' in node.meta:
                    node.target = node.meta['orig_target']
                    del node.meta['orig_target']
        model.recompile()
        model.graph.lint()


def update_internal_dict(module):
    prefix = module._hf_hook.weights_map.prefix
    for key in module.state_dict().keys():
        module._hf_hook.weights_map.dataset.state_dict[prefix + key] = recurse_getattr(
            module, key + '.data')


def offload_model(model, device='cuda'):
    # FX vs non-FX model need different offloading
    model = maybe_offload_weights_to_cpu(model, is_fx=hasattr(model, 'graph'))

    # We attach these functions to the hooked modules for convenience when modifying parameters
    # during PTQ (e.g. SmoothQuant).
    def allocate_params(module, device='cpu'):
        """
        This function calls the pre_forward function of the _hf_hook, making sure parameters are on
        the selected device, rather than on the meta device.
        """
        if module._hf_hook.offload == False:
            return
        module._hf_hook.pre_forward(module)
        # When quantizing and retrieving parameters (e.g., during GPTQ), we want to recurse through
        # all the submodules
        for m in module.modules():
            if hasattr(m, '_hf_hook'):
                m._hf_hook.pre_forward(m)

    def offload_params(module):
        """
        This functions moves the parameters back to the meta device, after making sure to update the
        internal state dict with the most recent values
        """
        if module._hf_hook.offload == False:
            return
        update_internal_dict(module)
        module._hf_hook.post_forward(module, torch.tensor([]))
        for m in module.modules():
            if hasattr(m, '_hf_hook'):
                m._hf_hook.post_forward(m, torch.tensor([]))

    for module in model.modules():
        if hasattr(module, '_hf_hook'):
            module.allocate_params = allocate_params
            module.offload_params = offload_params

    return model


# Adapted from optimum/exporters/onnx/convert.py, check_dummy_inputs_are_allowed
def get_forward_signature(model):
    from inspect import signature
    forward = model.forward if isinstance(model, torch.nn.Module) else model.call
    forward_parameters = signature(forward).parameters
    forward_inputs_set = set(forward_parameters.keys())
    return forward_inputs_set


def generate_dummy_inputs(self, forward_signature_keys=['input_ids'], verbose=False, **kwargs):
    dummy_inputs = self._brv_generate_dummy_inputs_orig(**kwargs)
    keys_to_delete = []
    for k in dummy_inputs.keys():
        if not k in forward_signature_keys:
            keys_to_delete.append(k)
    for k in keys_to_delete:
        if verbose:
            print(f"Deleting '{k}'")
        del dummy_inputs[k]
    return dummy_inputs
