# BNN-PYNQ Brevitas experiments

This repo contains training scripts and pretrained models to recreate the LFC and CNV models
used in the [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ) repo using [Brevitas](https://github.com/Xilinx/brevitas).
These pretrained models and training scripts are courtesy of 
[Alessandro Pappalardo](https://github.com/volcacius) and [Ussama Zahid](https://github.com/ussamazahid96).

## Experiments

| Name     | Input quantization           | Weight quantization | Activation quantization | Dataset       | Top1 accuracy |
|----------|------------------------------|---------------------|-------------------------|---------------|---------------|
| TFC_1W1A | 1 bit                        | 1 bit               | 1 bit                   |  MNIST        |    93.17%     |
| TFC_1W2A | 2 bit                        | 1 bit               | 2 bit                   |  MNIST        |    94.79%     |
| TFC_2W2A | 2 bit                        | 2 bit               | 2 bit                   |  MNIST        |    96.60%     |
| SFC_1W1A | 1 bit                        | 1 bit               | 1 bit                   |  MNIST        |    97.81%     |
| SFC_1W2A | 2 bit                        | 1 bit               | 2 bit                   |  MNIST        |    98.31%     |
| SFC_2W2A | 2 bit                        | 2 bit               | 2 bit                   |  MNIST        |    98.66%     |
| LFC_1W1A | 1 bit                        | 1 bit               | 1 bit                   |  MNIST        |    98.88%     |
| LFC_1W2A | 2 bit                        | 1 bit               | 2 bit                   |  MNIST        |    98.99%     |
| CNV_1W1A | 8 bit                        | 1 bit               | 1 bit                   |  CIFAR10      |    84.22%     |
| CNV_1W2A | 8 bit                        | 1 bit               | 2 bit                   |  CIFAR10      |    87.80%     |
| CNV_2W2A | 8 bit                        | 2 bit               | 2 bit                   |  CIFAR10      |    89.03%     |

## Train

A few notes on training:
- An experiments folder at */path/to/experiments* must exist before launching the training.
- Training is set to 1000 epochs for 1W1A networks, 500 otherwise. 
- Force-enabling the Pytorch JIT with the env flag PYTORCH_JIT=1 significantly speeds up training.

To start training a model from scratch, e.g. LFC_1W1A, run:
 ```bash
PYTORCH_JIT=1 brevitas_bnn_pynq_train --network LFC_1W1A --experiments /path/to/experiments
 ```

## Evaluate

To evaluate a pretrained model, e.g. LFC_1W1A, run:
 ```bash
PYTORCH_JIT=1 brevitas_bnn_pynq_train --evaluate --network LFC_1W1A --pretrained
 ```

To evaluate your own checkpoint, of e.g. LFC_1W1A, run:
 ```bash
PYTORCH_JIT=1 brevitas_bnn_pynq_train --evaluate --network LFC_1W1A --resume /path/to/checkpoint.tar
 ```
