import imp
import os
import torch
import sys
from packaging import version
import docrep

torch_version = version.parse(torch.__version__)

docstrings = docrep.DocstringProcessor()
# If pytorch version >= 1.3.0, it means that we have the compiled version of the functions that use the
# Straight-Trough-Estimator. If that is the case, we need to load those operations in torch.ops.
if torch_version >= version.parse("1.3.0"):
    lib_dir = os.path.dirname(__file__)
    _, path, _ = imp.find_module("_C", [lib_dir])
    torch.ops.load_library(path)
