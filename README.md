# Brevitas

[![Downloads](https://pepy.tech/badge/brevitas)](https://pepy.tech/project/brevitas)
[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a PyTorch library for neural network quantization, with support for both *post-training quantization (PTQ)* and *quantization-aware training (QAT)*.

**Please note that Brevitas is a research project and not an official Xilinx product.**

If you like this project please consider ⭐ this repo, as it is the simplest and best way to support it.

If you have issues, comments, or are just looking for advices on training quantized neural networks, open an issue or a discussion.

## Cite as

If you adopt Brevitas in your work, please cite it as:
```
@software{brevitas,
  author       = {Alessandro Pappalardo},
  title        = {Xilinx/brevitas},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3333552},
  url          = {https://doi.org/10.5281/zenodo.3333552}
}
```


## History

- *2023/04/21* - Release version 0.9.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.9.0).
- *2023/01/10* - Release version 0.8.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.8.0).
- *2021/12/13* - Release version 0.7.1, fix a bunch of issues. Added TVMCon 2021 tutorial notebook.
- *2021/11/03* - Re-release version 0.7.0 (build 1) on PyPI to fix a packaging issue.
- *2021/10/29* - Release version 0.7.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.7.0).
- *2021/06/04* - Release version 0.6.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.6.0).
- *2021/05/24* - Release version 0.5.1, fix a bunch of minor issues. See [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.5.1).
- *2021/05/06* - Release version 0.5.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.5.0).
- *2021/03/15* - Release version 0.4.0, add support for \_\_torch_function\_\_ to QuantTensor.
- *2021/03/04* - Release version 0.3.1, fix bug w/ act initialization from statistics w/ IGNORE_MISSING_KEYS=1.
- *2021/03/01* - Release version 0.3.0, implements enum and shape solvers within extended dependency injectors. This allows declarative quantizers to be self-contained.
- *2021/02/04* - Release version 0.2.1, includes various bugfixes of QuantTensor w/ zero-point.
- *2021/01/30* - First release version 0.2.0 on PyPI.

## Requirements

* Python >= 3.7 .
* [Pytorch](https://pytorch.org) >= 1.5.1 .
* Windows, Linux or macOS.
* GPU training-time acceleration (*Optional* but recommended).

## Installation

You can install the latest release from PyPI:
```bash
pip install brevitas
```

## Getting started

Check out available info at https://xilinx.github.io/brevitas/getting_started .
