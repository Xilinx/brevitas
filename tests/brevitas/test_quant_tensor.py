import torch

from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor


def test_qt_structure():
    qt = IntQuantTensor(
        torch.randn(10), torch.randn(1), torch.tensor(0.), torch.tensor(8.), True, False)
    assert isinstance(qt, IntQuantTensor)
    assert isinstance(qt, QuantTensor)
    assert isinstance(qt, tuple)
    assert hasattr(qt, '_fields')
    assert len(qt._fields) == 6
