# Post Training Quantization

This folder shows how to use Brevitas PTQ features to quantize torchvision models, as well as how to apply PTQ to models that have been manually defined with Brevitas quantized layers (such as MobileNet V1).
The flow presented can potentially be adopted as an entry point to QAT for additionally finetuning, especially at lower precisions.


We provide two workflows:
- An evaluation script that, given a model name and a quantization configuration, performs PTQ on the model, evaluates its ImageNet top1 accuracy, and optionally exports it to either ONNX or TorchScript QCDQ.
- A benchmark suite that tests several quantization configurations on a few selected models.

Three types of target backend are exposed for programmatic quantization. Different backends dictate different structural policies for how a network should be quantized:
- *fx*:
  - The number of re-quantization ops is minimized by re-quantizing only when necessary, avoiding consecutive quantization ops if possible.
  - Adds are quantized to have the same scale at the input, but allows for different signs.
  - Concats are quantized to have the same scale, zero-point, sign and bit-width.
  - Activation quantization is performed earlier rather than later, so e.g. activation functions are quantized at their output.
  - Activations are quantized to unsigned when possible during symmetric quantization (e.g. ReLU).
- *layerwise* - Quantizes only the input and the weights to compute-heavy layers. Other layers are left unquantized.
- *flexml* - An internal backend.


The implementation for programmatic quantization is still experimental and might break on certain input models with certain configurations.


For both these flows, the following options are exposed:
- Bit-width of weight and activations.
- In case of minifloat quantization, the exponent and mantissa bit-width of weights and activations.
- Scales can be either float32 or power-of-two (po2) numbers.
- Weights' scale factors can be either per-tensor or per-channel.
- Biases can be floating point, int16, or int32.
- Activation quantization can be symmetric or asymmetric.
- Possibility to use statistics or MSE for scale factor computations for weights and activations.
- Percentiles used for the activations' statistics computation during calibration.

Furthermore, Brevitas additional PTQ techniques can be enabled:
- Bias correction[<sup>1 </sup>].
- Graph equalization[<sup>1 </sup>].
- If Graph equalization is enabled, the _merge\_bias_ technique can be enabled.[<sup>2 </sup>] [<sup>3 </sup>].
- GPTQ [<sup>4 </sup>].
- Learned Round [<sup>5 </sup>].
- GPFQ [<sup>6 </sup>].
- Channel splitting [<sup>7 </sup>].
- Activation Equalization [<sup>8 </sup>].


Internally, when defining a quantized model programmatically, Brevitas leverages `torch.fx` and its `symbolic_trace` functionality, meaning that an input model is required to pass symbolic tracing for it to work.

For more information about what are the currently supported quantized layers in Brevitas, check the [following file](https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/quantize.py),
where we map the torch compute layers and activations with their corresponding quantized version.

Pre-computed accuracy results for torchvision can be found under `RESULTS_TORCHVISION.csv` with several quantization combinations evaluated on three different torchvision models (ResNet18, MobileNet V2, ViT B32), while results on the hand defined quantized MobileNet V1 can be found under `RESULTS_IMGCLSMOB.csv`.
Furthemore, for the torchvision models, we provide a `RESULTS_TORCHVISION_BEST_CONFIGS.csv`, where, for each combination of hardware-related features (e.g., support of per-channel scaling factors), we report the best configurations and their results.

## Requirements

To install all the necessary requirements, simply run:

```bash
pip install brevitas[vision]
```

## Evaluation flow

This flow allows to specify which pre-trained torchvision model to quantize and apply PTQ to with the desired quantization configuration.
It also gives the possibility to export the model to either ONNX QCDQ format or in torch QCDQ format.
The quantization and export options to specify are:

```bash
{{ readme_help }}
```

The script requires to specify the calibration folder (`--calibration-dir`), from which the calibration samples will be taken (configurable with the `--calibration-samples` argument), and a validation folder (`--validation-dir`).

For example, to run the script on the GPU 0:
```bash
brevitas_ptq_imagenet_val --calibration-dir /path/to/imagenet/calibration/folder --validation-dir /path/to/imagenet/validation/folder --gpu 0
```


[<sup>1 </sup>]: https://arxiv.org/abs/1906.04721
[<sup>2 </sup>]: https://github.com/Xilinx/Vitis-AI/blob/50da04ddae396d10a1545823aca30b3abb24a276/src/vai_quantizer/vai_q_pytorch/nndct_shared/optimization/commander.py#L450
[<sup>3 </sup>]: https://github.com/openppl-public/ppq/blob/master/ppq/quantization/algorithm/equalization.py
[<sup>4 </sup>]: https://arxiv.org/abs/2210.17323
[<sup>5 </sup>]: https://arxiv.org/abs/2004.10568
[<sup>6 </sup>]: https://arxiv.org/abs/2201.11113
[<sup>7 </sup>]: https://arxiv.org/abs/1901.09504
[<sup>8 </sup>]: https://arxiv.org/abs/2211.10438
