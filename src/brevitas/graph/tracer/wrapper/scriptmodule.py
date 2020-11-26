from dataclasses import replace

import torch
from torch.nn import Module

from brevitas.graph.module import FnType, _replace_module


def torchscript_wrapper(model):
    script_modules = set([])
    names, _ = zip(*list(model.named_modules()))
    prefix_list_list = [name.split(".") for name in names]
    for prefix_list in prefix_list_list:
        supermodule = model
        for prefix in prefix_list:
            if prefix != '':
                submodule = supermodule._modules[prefix]
                if not isinstance(supermodule, torch.jit.ScriptModule) \
                        and isinstance(submodule, torch.jit.ScriptModule):
                    script_modules.add(submodule)
                    break
                else:
                    supermodule = submodule
    for sm in script_modules:
        _replace_module(model, sm, ScriptModuleWrapper(sm))


class ScriptModuleWrapper(Module):

    def __init__(self, script_module):
        super().__init__()
        self.script_module = script_module

    def forward(self, x):  # TODO multiple inputs
        assert not isinstance(x, tuple), 'Multiple inputs not supported'
        if isinstance(x, Tracer):
            out = self.script_module(x.tensor_)
            if isinstance(out, tuple):
                out = [replace(x, tensor_=o) for o in out]  # they all have to be tensors
                x._update_trace(self.script_module, FnType.SCRIPTMODULE, [x.tensor_], {}, out)
            return out[0]  # TODO multiple outputs
        else:
            return self.script_module(x)