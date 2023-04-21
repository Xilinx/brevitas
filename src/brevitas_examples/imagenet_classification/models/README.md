# Quantized model definitions

## MobileNet V1

This low-precision MobileNet V1 implementation makes the following assumptions:
- Floating point per-channel scales.
- Input preprocessing is modified to have a single scale factor rather than a per-channel one, so that it can be propagated through the first convolution.
- Weights of the first layer are always quantized to 8 bit.
- Padding in the first convolution is removed, so that the input's mean can be propagated through the first convolution to thresholds.
- Biases and batch-norm can be merged with scales into requantization ops, and as such as left unquantized. The only exception is the bias of the fully connected layer, which is quantized.
- Scaling of the fully connected layer is per-layer, so that the output of the network doesn't require rescaling.
- Per-channel scale for the inputs to depthwise sperable convolution layers, which can be propagated through the convolution.
- Quantized avg pool performs a sum followed by truncation or rounding to the specified bit-width (in place of a division).

## ProxylessNAS Mobile14

This low-precision ProxylessNAS Mobile14 (a MobileNet V2 *style* type of network) implementation makes the following assumptions:
- Floating point per-channel scales.
- Input preprocessing is modified to have a single scale factor rather than a per-channel one, so that it can be propagated through the first convolution.
- Biases and batch-norm can be merged with scales into requantization ops, and as such as left unquantized. The only exception is the bias of the fully connected layer, which is quantized.
- Scaling of the fully connected layer is per-layer, so that the output of the network doesn't require rescaling.
- Per-channel scale for the inputs to depthwise sperable convolution layers, which can be propagated through the convolution.
- Quantized avg pool performs a sum followed by a truncation to the specified bit-width (in place of a division).
- An optional Hadamard layer as the final classification layer, in place of a QuantLinear layer.
- Quantized avg pool performs a sum followed by truncation or rounding to the specified bit-width (in place of a division).

## VGG

This low-precision implementation of VGG makes the following assumptions:
- Floating point per-channel scale factors.
- Biases and batch-norm are left unquantized.
- Quantized avg pool performs a sum followed by rounding to the specified bit-width (in place of a division).
