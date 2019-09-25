# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

## VGG

The reduced-precision implementation of VGG makes the following assumptions:
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Biases and batch-norm can be merged into FINN-style thresholds, and as such as left unquantized.
- Quantizing avg pooling requires to propagate scaling factors along the forward pass, which generates some additional verbosity.
  To keep things simple, this particular example then leaves avg pooling unquantized. 