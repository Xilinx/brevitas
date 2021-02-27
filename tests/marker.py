import pytest
from packaging.version import parse

from brevitas import torch_version


def requires_pt_ge(pt_version: str):
    skip = parse(pt_version) > torch_version

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch >= {pt_version}')(f)

    return skip_wrapper


def requires_pt_lt(pt_version: str):
    skip = parse(pt_version) <= torch_version

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch < {pt_version}')(f)

    return skip_wrapper