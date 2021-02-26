from platform import system
import pytest
from packaging.version import parse
import torch
import numpy

# Fails with Windows under Nox + 1.18+ numpy
PT_VERSION = parse(torch.__version__)
NUMPY_VERSION_TO_SKIP = parse(numpy.__version__) >= parse('1.18')
IS_WIN = system() == 'Windows'
WIN_NOX_REASON = 'Known issue with Nox and Numpy 1.18+ on Windows'
skip_win_nox_numpy = pytest.mark.skipif(
    NUMPY_VERSION_TO_SKIP and IS_WIN, reason=WIN_NOX_REASON)


def requires_pt_ge(pt_version: str):
    skip = parse(pt_version) > PT_VERSION

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch >= {pt_version}')(f)

    return skip_wrapper


def requires_pt_lt(pt_version: str):
    skip = parse(pt_version) <= PT_VERSION

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch < {pt_version}')(f)

    return skip_wrapper