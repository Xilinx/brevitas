import imp
import os
import torch
import docrep
from brevitas.config import MIN_TORCH_JITTABLE_VERSION, MAX_TORCH_JITTABLE_VERSION
import torch
from packaging import version
import brevitas.jit as jit

torch_version = version.parse(torch.__version__)
IS_STE_JITTABLE = version.parse(MIN_TORCH_JITTABLE_VERSION) <= torch_version <= version.parse(MAX_TORCH_JITTABLE_VERSION)

docstrings = docrep.DocstringProcessor()

# If pytorch version >= 1.3.0, it means that we have the compiled version of the functions that use the
# Straight-Trough-Estimator. If that is the case, we need to load those operations in torch.ops.
if IS_STE_JITTABLE:
    lib_dir = os.path.dirname(__file__)
    _, path, _ = imp.find_module("_C", [lib_dir])
    torch.ops.load_library(path)


