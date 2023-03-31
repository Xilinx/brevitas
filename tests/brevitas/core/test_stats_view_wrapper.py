import warnings

import pytest
import torch
import torch.nn as nn

from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.view_wrapper import _ViewCatParameterWrapper
from brevitas.core.stats.view_wrapper import _ViewParameterWrapper


def test_scaling_state_dict_viewparameterwrapper():

    class TestModuleStd(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)
            self.scaling_op = _ViewParameterWrapper(self.conv.weight, nn.Identity())

    mod = TestModuleStd()

    with warnings.catch_warnings(record=True) as wlist:
        mod.state_dict()
        for w in wlist:
            assert "Positional args are being deprecated" not in str(w.message)


def test_scaling_state_dict_viewcatparameterwrapper():

    class TestModuleCat(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)
            self.scaling_op = _ViewCatParameterWrapper(self.conv.weight, nn.Identity(), 1)

    mod = TestModuleCat()

    with warnings.catch_warnings(record=True) as wlist:
        mod.state_dict()
        for w in wlist:
            assert "Positional args are being deprecated" not in str(w.message)
