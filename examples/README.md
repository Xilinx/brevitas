# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

Below in the table is a list of example pretrained models made available for reference.

|   | Name         | Scaling Type               | First layer weights | Weights | Activations | Avg pool | Top1  | Top5  | Pretrained model                                                                                | Retrained from                                                |
|---|--------------|----------------------------|---------------------|---------|-------------|----------|-------|-------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
|   | MobileNet V1 | Floating-point per channel | 8 bit               | 4 bit   | 4 bit       | 4 bit    | 71.25 | 90.10 | [Download](https://github.com/Xilinx/brevitas/releases/download/examples-0.0.1/mobilenetv1.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |

## MobileNet V1

The reduced-precision implementation of MobileNet V1 makes the following assumptions:
- Input preprocessing is not modified compared to the full-precision model.
- Weights of the first layer are always quantized to 8 bit.
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Biases and batch-norm can be merged into FINN-style thresholds, and as such as left unquantized. The only exception is the bias of the fully connected layer, which is quantized.
- Scaling of the fully connected layer is per-layer, so that the output of the network doesn't require rescaling.
- Per-channel scale factors before depthwise convolution layers can be propagate through the convolution.
- Quantized avg pool performs a sum followed by a truncation to the specified bit-width (in place of a division).

To evaluate the quantized pretrained model on ImageNet:

- Download the pretrained model linked in the table above.
- Within a Pytorch environment with Brevitas installed and the ImageNet dataset in a Pytorch friendly format, run:

```
python imagenet_val.py --resume /path/to/mobilenetv1.pth --bit-width 4 --imagenet-dir /path/to/imagenet
```

## VGG

The reduced-precision implementation of VGG makes the following assumptions:
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Biases and batch-norm can be merged into FINN-style thresholds, and as such as left unquantized.
- Quantizing avg pooling requires to propagate scaling factors along the forward pass, which generates some additional verbosity.
  To keep things simple, this particular example then leaves avg pooling unquantized. 