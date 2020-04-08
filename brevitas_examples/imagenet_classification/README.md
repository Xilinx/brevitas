# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

Below in the table is a list of example pretrained models made available for reference.

| Name         | Cfg                   | Scaling Type               | First layer weights | Weights | Activations | Avg pool | Top1  | Top5  | Pretrained model                                                                                | Retrained from                                                |
|--------------|-----------------------|----------------------------|---------------------|---------|-------------|----------|-------|-------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| MobileNet V1 | quant_mobilenet_v1_4b | Floating-point per channel | 8 bit               | 4 bit   | 4 bit       | 4 bit    | 71.14 | 90.10 | [Download](https://github.com/Xilinx/brevitas/releases/download/quant_mobilenet_v1_4b-r1/quant_mobilenet_v1_4b-0100a667.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |
| ProxylessNAS Mobile14 w/ Hadamard classifier | quant_proxylessnas_mobile14_hadamard_4b  | Floating-point per channel | 8 bit               | 4 bit  | 4 bit       | 4 bit    | 73.52 | 91.46 | [Download](https://github.com/Xilinx/brevitas/releases/download/quant_proxylessnas_mobile14_hadamard_4b-r0/quant_proxylessnas_mobile14_hadamard_4b-4acbfa9f.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |
| ProxylessNAS Mobile14 | quant_proxylessnas_mobile14_4b | Floating-point per channel | 8 bit               | 4 bit  | 4 bit       | 4 bit    | 74.42 | 92.04 | [Download](https://github.com/Xilinx/brevitas/releases/download/quant_proxylessnas_mobile14_4b-r0/quant_proxylessnas_mobile14_4b-e10882e1.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |
| ProxylessNAS Mobile14 | quant_proxylessnas_mobile14_4b5b | Floating-point per channel | 8 bit               | 4 bit, 5 bit  | 4 bit, 5 bit       | 4 bit    | 75.01 | 92.33 | [Download](https://github.com/Xilinx/brevitas/releases/download/quant_proxylessnas_mobile14_4b5b-r0/quant_proxylessnas_mobile14_4b5b-2bdf7f8d.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |


To evaluate a pretrained quantized model on ImageNet:
 
 - Make sure you have Brevitas installed and the ImageNet dataset in a Pytorch friendly format (following this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)).
 - Pass the name of the model as an input to the evaluation script. The required checkpoint will be downloaded automatically. 
 
 For example, for *quant_mobilenet_v1_4b* evaluated on GPU 0:

```
brevitas_imagenet_val --imagenet-dir /path/to/imagenet --model quant_mobilenet_v1_4b --gpu 0 --pretrained
```

## MobileNet V1

The reduced-precision implementation of MobileNet V1 makes the following assumptions:
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Input preprocessing is modified to have a single scale factor rather than a per-channel one, so that it can be propagated through the first convolution to thresholds.
- Weights of the first layer are always quantized to 8 bit.
- Padding in the first convolution is removed, so that the input's mean can be propagated through the first convolution to thresholds.
- Biases and batch-norm can be merged into FINN-style thresholds, and as such as left unquantized. The only exception is the bias of the fully connected layer, which is quantized.
- Scaling of the fully connected layer is per-layer, so that the output of the network doesn't require rescaling.
- Per-channel scale factors before depthwise convolution layers can be propagate through the convolution.
- Quantized avg pool performs a sum followed by a truncation to the specified bit-width (in place of a division).

## VGG

The reduced-precision implementation of VGG makes the following assumptions:
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Biases and batch-norm can be merged into FINN-style thresholds, and as such as left unquantized.
- Quantizing avg pooling requires to propagate scaling factors along the forward pass, which generates some additional verbosity.
  To keep things simple, this particular example then leaves avg pooling unquantized. 
