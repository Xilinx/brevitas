import os
import glob
import warnings
from packaging import version
from pkg_resources import get_distribution, DistributionNotFound

from torch.utils import cpp_extension
import torch

import brevitas.jit as jit
from brevitas import config


pkg_dir = os.path.dirname(os.path.abspath(__file__))

if torch.__version__.endswith('+cpu'):
    torch_version = version.parse(torch.__version__.rstrip('+cpu'))
else:
    torch_version = version.parse(torch.__version__)

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
