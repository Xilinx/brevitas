import os
from platform import system
import pytest
from packaging import version
import torch
import numpy

# Fails with Windows under Nox + 1.18+ numpy
NUMPY_VERSION = version.parse(numpy.__version__) >= version.parse('1.18')
IS_WIN = system() == 'Windows'
WIN_NOX_REASON = 'Known issue with Nox and Numpy 1.18+ on Windows'
check_expected_win_nox_fail = pytest.mark.xfail(NUMPY_VERSION and IS_WIN, reason=WIN_NOX_REASON)

# Setup expected fail for Pytorch 1.1.0
PYT_120_JIT_CONDITION = version.parse(torch.__version__) == version.parse('1.1')
PYT_120_JIT_REASON = 'Known limitation of Pytorch 1.1.0'
check_expected_pyt_110_fail = pytest.mark.xfail(PYT_120_JIT_CONDITION, reason=PYT_120_JIT_REASON, raises=RuntimeError)

# Setup expected fail for Pytorch 1.2.0 and JIT Disabled
PYT_120_JIT_CONDITION = version.parse(torch.__version__) == version.parse('1.2') and os.environ.get('PYTORCH_JIT', '1') == '0'
PYT_120_JIT_REASON = 'Known bug to Pytorch 1.2.0 with JIT disabled'
check_expected_pyt_120_fail = pytest.mark.xfail(PYT_120_JIT_CONDITION, reason=PYT_120_JIT_REASON, raises=RuntimeError)

# Setup expected fail for mock and JIT Enabled for Pytorch < 1.4.0
MOCK_JIT_CONDITION = version.parse(torch.__version__) < version.parse('1.4') and os.environ.get('PYTORCH_JIT', '1') == '1'
MOCK_JIT_REASON = 'Cannot use Mock class with pytorch JIT enabled'
check_mock_jit_pyt_l140_fail = pytest.mark.xfail(MOCK_JIT_CONDITION, reason=MOCK_JIT_REASON, raises=AttributeError)

# Setup expected fail for mock and JIT Enabled for Pytorch >= 1.4.0
MOCK_JIT_CONDITION = version.parse(torch.__version__) >= version.parse('1.4') and os.environ.get('PYTORCH_JIT', '1') == '1'
MOCK_JIT_REASON = 'Cannot use Mock class with pytorch JIT enabled'
check_mock_jit_pyt_ge140_fail = pytest.mark.xfail(MOCK_JIT_CONDITION, reason=MOCK_JIT_REASON, raises=RuntimeError)

def combine_conditions(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco