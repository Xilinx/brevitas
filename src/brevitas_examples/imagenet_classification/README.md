# Imagenet Examples

This folder contains examples about how to leverage quantized layers and quantization flows offered by Brevitas for the task of ImageNet classification.

Brevitas allows to define quantized models in two different ways:
- *By hand*, writing a quantized model using ``brevitas.nn`` quantized layers, possibly by modifying an original PyTorch floating-point model definition.
- *Programmatically*, by taking a floating-point model as input and replacing `torch.nn` layers with `brevitas.nn` layers according to some user-defined criteria.

Both model definitions can then be used as a starting point for either PTQ (Post-Training Quantization), QAT (Quantization Aware Training), or PTQ followed by QAT.

The examples in this folder are organized in two sections:
- PTQ: Examples on how to use Brevitas PTQ features. We show how to apply PTQ to both models that have been defined by-hand with Brevitas quantized layers (by manually modifying a pretrained floating-point model definition), and to models that have been programmatically defined starting from their `torchvision` definition.
- QAT: Examples on how to run inference on a small set of networks that have been trained with QAT. For each model, the corresponding quantized model definition is also provided. The goal is to provide example low precision (< 8b) models that can be exported and leveraged for acceleration within downstream compilers (the whole training code is currently not available).

For more details, check the corresponding folders `ptq` and `qat`. The `models` folder contains quantized models defined by hand that are leveraged by both the PTQ and QAT examples. Checkout its README for more information on how ieach model is defined.
