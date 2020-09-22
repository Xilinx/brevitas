# Monkey patch Injector to avoid issues with interactive debugging
# Lack of __len__ magic on Injector breaks the debugger ability to inspect it
from _dependencies.injector import _InjectorType, __init__, let, injector_doc


class _ExtendedInjectorType(_InjectorType):

    def __len__(self):
        return None


Injector = _ExtendedInjectorType(
    "Injector",
    (),
    {"__init__": __init__, "__doc__": injector_doc, "let": classmethod(let)})


import _dependencies
_dependencies.injector.Injector = Injector

import imp
import os
import torch
import docrep
from brevitas.config import MIN_TORCH_JITTABLE_VERSION, MAX_TORCH_JITTABLE_VERSION
import torch
from packaging import version
torch_version = version.parse(torch.__version__)

is_ste_jittable = version.parse(MIN_TORCH_JITTABLE_VERSION) <= torch_version <= version.parse(MAX_TORCH_JITTABLE_VERSION)

docstrings = docrep.DocstringProcessor()

# If pytorch version >= 1.3.0, it means that we have the compiled version of the functions that use the
# Straight-Trough-Estimator. If that is the case, we need to load those operations in torch.ops.
if is_ste_jittable:
    lib_dir = os.path.dirname(__file__)
    _, path, _ = imp.find_module("_C", [lib_dir])
    torch.ops.load_library(path)
