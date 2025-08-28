# Brevitas

[![Downloads](https://pepy.tech/badge/brevitas)](https://pepy.tech/project/brevitas)
[![Pytest](https://github.com/Xilinx/brevitas/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/Xilinx/brevitas/actions/workflows/pytest.yml)
[![Examples Pytest](https://github.com/Xilinx/brevitas/actions/workflows/examples_pytest.yml/badge.svg?branch=master)](https://github.com/Xilinx/brevitas/actions/workflows/examples_pytest.yml)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a PyTorch library for neural network quantization, with support for both *post-training quantization (PTQ)* and *quantization-aware training (QAT)*.

**Please note that Brevitas is a research project and not an official Xilinx product.**

If you like this project please consider ⭐ this repo, as it is the simplest and best way to support it.

## Requirements

* Python >= 3.9, <3.13
* [Pytorch](https://pytorch.org) >= 1.12, <= 2.8 (more recent versions would be untested).
* Windows, Linux or macOS.
* GPU training-time acceleration (*Optional* but recommended).

## Installation

You can install the latest release from PyPI:
```bash
pip install brevitas
```

## Getting Started

Brevitas currently offers quantized implementations of the most common PyTorch layers used in DNN under `brevitas.nn`, such as `QuantConv1d`, `QuantConv2d`, `QuantConvTranspose1d`, `QuantConvTranspose2d`, `QuantMultiheadAttention`, `QuantRNN`, `QuantLSTM` etc., for adoption within PTQ and/or QAT.
For each one of these layers, quantization of different tensors (inputs, weights, bias, outputs, etc) can be individually tuned according to a wide range of quantization settings.

As a reference for PTQ, Brevitas provides an example user flow for ImageNet classification models under [`brevitas_examples.imagenet_classification.ptq`](https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/imagenet_classification/ptq/ptq_evaluate.py) that quantizes an input torchvision model using PTQ under different quantization configurations (e.g. bit-width, granularity of scale, etc).

For more info, checkout our [documentation](https://xilinx.github.io/brevitas/).

## Cite as

If you adopt Brevitas in your work, please cite it as:
```
@software{brevitas,
  author       = {Franco, Giuseppe and Pappalardo, Alessandro and Fraser, Nicholas J},
  title        = {Xilinx/brevitas},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3333552},
  url          = {https://doi.org/10.5281/zenodo.3333552}
}
```

## History

- *2025/08/28* - Release version 0.12.1, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.12.1).
- *2025/05/09* - Release version 0.12.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.12.0).
- *2024/10/10* - Release version 0.11.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.11.0).
- *2024/07/23* - Minor release version 0.10.3, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.10.3).
- *2024/02/19* - Minor release version 0.10.2, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.10.2).
- *2024/02/15* - Minor release version 0.10.1, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.10.1).
- *2023/12/08* - Release version 0.10.0, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.10.0).
- *2023/04/28* - Minor release version 0.9.1, see the [release notes](https://github.com/Xilinx/brevitas/releases/tag/v0.9.1).
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
