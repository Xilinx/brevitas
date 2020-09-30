from packaging import version

from brevitas import torch_version


if torch_version >= version.parse('1.3.0'):
    OPSET = 11
else:
    OPSET = 10