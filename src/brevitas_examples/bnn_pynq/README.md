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
| RESNET18_4W4A | 8 bit (assumed)         | 4 bit               | 4 bit                   |  CIFAR10      |    92.61%     |

## Train

A few notes on training:
- An experiments folder at */path/to/experiments* must exist before launching the training.
- Set training to 1000 epochs for 1W1A networks, 500 otherwise with the `--epochs` flag.
- Enabling the JIT with the env flag BREVITAS_JIT=1 significantly speeds up training.

To start training a model from scratch, e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python bnn_pynq_train.py --network LFC_1W1A --experiments /path/to/experiments
 ```

## Evaluate

To evaluate a pretrained model, e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python bnn_pynq_train.py --evaluate --network LFC_1W1A --pretrained
 ```

To evaluate your own checkpoint, of e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python bnn_pynq_train.py --evaluate --network LFC_1W1A --resume /path/to/checkpoint.tar
 ```

## ONNX Export

The models can be exported to either QONNX or ONNX QCDQ by adding the `--export_qonnx`, `--export_qcdq_onnx` flags respectively.
This flag can be added to any training or evaluation run to export the model at the end of the process.
Note, to export an ONNX model,
Brevitas' JIT must be disabled (i.e., `BREVITAS_JIT=0`),
so it may be convenient to export the ONNX model as separate step _after_ training.

For example, you may want train as described [above](#train),
then export as a separate step as follows:

```bash
BREVITAS_JIT=0 python bnn_pynq_train.py --evaluate --network LFC_1W1A --resume /path/to/checkpoint.tar --export_qonnx
```

Note, ONNX export also works with pretrained models as follows:

```bash
BREVITAS_JIT=0 python bnn_pynq_train.py --evaluate --network LFC_1W1A --pretrained --export_qonnx --experiments /path/to/export_dir
```
