import os
import glob
import warnings

import docrep
from torch.utils import cpp_extension

import brevitas.jit as jit
from brevitas import config

docstrings = docrep.DocstringProcessor()

extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
sources = glob.glob(os.path.join(extensions_dir, '*.cpp'))
sources = [os.path.join(extensions_dir, s) for s in sources]

if config.NATIVE_STE_BACKEND_ENABLED:
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
