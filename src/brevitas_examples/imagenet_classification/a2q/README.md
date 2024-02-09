# Integer-Quantized Image Classification Models Trained on CIFAR10 with Brevitas

This directory contains scripts demonstrating how to train integer-quantized image classification models using accumulator-aware quantization (A2Q) as proposed in our ICCV 2023 paper "[A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance](https://arxiv.org/abs/2308.13504)".
Code is also provided to demonstrate A2Q+ as proposed in our arXiv paper "[A2Q+: Improving Accumulator-Aware Weight Quantization](https://arxiv.org/abs/2401.10432)", where we introduce the zero-centered weight quantizer (i.e., `AccumulatorAwareZeroCenterWeightQuant`) as well as the Euclidean projection-based weight initialization (EP-init).

## Experiments

All models are trained on the CIFAR10 dataset.
Input images are normalized to have unit mean and variance.
During training, random cropping is applied, along with random horizontal flips.
All residual connections are quantized to the specified activation bit width.


| Model Name | Weight Quantization | Activation Quantization | Target Accumulator | Top-1 Accuracy (%) |
|-----------------------------|----------------|---------------------|-------------------------|----------------------------|
| [float_resnet18](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/float_resnet18-1d98d23a.pth) | float32 | float32 | float32 | 95.0 |
||
| [quant_resnet18_w4a4_a2q_16b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_16b-d0af41f1.pth) | int4 | uint4 | int16 | 94.2 |
| [quant_resnet18_w4a4_a2q_15b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_15b-0d5bf266.pth) | int4 | uint4 | int15 | 94.2 |
| [quant_resnet18_w4a4_a2q_14b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_14b-267f237b.pth) | int4 | uint4 | int14 | 92.6 |
| [quant_resnet18_w4a4_a2q_13b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_13b-8c31a2b1.pth) | int4 | uint4 | int13 | 89.8 |
| [quant_resnet18_w4a4_a2q_12b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_12b-8a440436.pth) | int4 | uint4 | int12 | 83.9 |
||
| [quant_resnet18_w4a4_a2q_plus_16b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_16b-19973380.pth) | int4 | uint4 | int16 | 94.2 |
| [quant_resnet18_w4a4_a2q_plus_15b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_15b-3c89551a.pth) | int4 | uint4 | int15 | 94.1 |
| [quant_resnet18_w4a4_a2q_plus_14b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_14b-5a2d11aa.pth) | int4 | uint4 | int14 | 94.1 |
| [quant_resnet18_w4a4_a2q_plus_13b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_13b-332aaf81.pth) | int4 | uint4 | int13 | 92.8 |
| [quant_resnet18_w4a4_a2q_plus_12b](https://github.com/Xilinx/brevitas/releases/download/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_12b-d69f003b.pth) | int4 | uint4 | int12 | 90.6 |
