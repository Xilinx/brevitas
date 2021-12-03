import os
import glob
import warnings
from packaging import version
from pkg_resources import get_distribution, DistributionNotFound
from typing import List, Optional

from torch.utils import cpp_extension
import torch
from torch import Tensor

import brevitas.jit as jit
from brevitas import config


pkg_dir = os.path.dirname(os.path.abspath(__file__))

if torch.__version__.endswith('+cpu'):
    torch_version = version.parse(torch.__version__.rstrip('+cpu'))
else:
    torch_version = version.parse(torch.__version__)


if torch_version < version.parse('1.7.0'):
    from torch._overrides import has_torch_function, handle_torch_function
    original_cat = torch.cat

    @torch.jit.ignore
    def unsupported_jit_cat(tensors, dim):
        if not isinstance(tensors, (tuple, list)):
            tensors = tuple(tensors)
            return unsupported_jit_cat(tensors, dim)
        if any(type(t) is not Tensor for t in tensors) and has_torch_function(tensors):
            return handle_torch_function(
                original_cat, relevant_args=tensors, tensors=tensors, dim=dim)
        else:
            return original_cat(tensors=tensors, dim=dim)

    def cat(
            tensors: List[Tensor],
            dim: int = 0) -> Tensor:
        if not torch.jit.is_scripting():
            return unsupported_jit_cat(tensors, dim)
        return original_cat(tensors, dim=dim)

    torch.cat = cat


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


if config.JIT_ENABLED:
    extensions_dir = os.path.join(pkg_dir, 'csrc')
    sources = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    sources = [os.path.join(extensions_dir, s) for s in sources]

    try:
        cpp_extension.load(
            name='autograd_ste_ops',
            sources=sources,
            is_python_module=False,
            verbose=config.VERBOSE)
        NATIVE_STE_BACKEND_LOADED = True
    except:
        warnings.warn("Brevitas' native STE backend is enabled but couldn't be loaded")
        NATIVE_STE_BACKEND_LOADED = False
else:
    NATIVE_STE_BACKEND_LOADED = False
