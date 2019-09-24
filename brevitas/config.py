import os
import docrep

docstrings = docrep.DocstringProcessor()

IGNORE_MISSING_KEYS = bool(os.environ.get('BREVITAS_IGNORE_MISSING_KEYS', False))
REINIT_WEIGHT_QUANT_ON_LOAD = bool(os.environ.get('BREVITAS_REINIT_WEIGHT_QUANT_ON_LOAD', True))