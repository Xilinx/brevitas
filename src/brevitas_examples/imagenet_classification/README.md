# Imagenet Examples

This folder contains examples about how to leverage quantized layers and quantization flows offered by Brevitas.

There are two main category of examples at the moment:
- QAT (Quantization Aware Training): Examples on how to run inference on a small set of pre-trained quantized networks, obtained through QAT. For each model, the corresponding quantized model definition is also provided.
- PTQ (Post-Training Quantization): Examples on how to use the Brevitas PTQ flow. Currently we support the possibility to calibrate models that have been defined with Brevitas quantized modules, as well as automatically quantizing and calibrating models coming from torchvision.

For more details, check the corresponding folders.
