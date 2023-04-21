# Post Training Quantization

This folder shows how to use Brevitas PTQ features to quantize torchvision models, as well as how to apply PTQ to models that have been manually defined with Brevitas quantized layers (such as MobileNet V1).
The flow presented can potentially be adopted as an entry point to QAT for additionally finetuning, especially at lower precisions.


We provide two workflows:
- An evaluation script that, given a model name and a quantization configuration, performs PTQ on the model, evaluates its ImageNet top1 accuracy, and optionally exports it to either ONNX or TorchScript QCDQ.
- A benchmark suite that tests several quantization configurations on a few selected models.

Three types of target backend are exposed for programmatic quantization. Different backends dictate different structural policies for how a network should be quantized:
- `generic`
  - The number of re-quantization ops is minimized by re-quantizing only when necessary, avoiding consecutive quantization ops if possible.
  - Adds are quantized with the scale at the input, but allows for different signs.
  - Concats are quantized to have the same scale, zero-point, sign and bit-width.
  - Activation quantization is performed earlier rather than later, so e.g. activation functions are quantized at their output.
  - Activations are quantized to unsigned when possible, even with symmetric quantization.
- `layerwise` - Quantizes only the input and the weights to compute-heavy layers. Other layers are left unquantized.
- `flexml` - An internal backend.
The implementation for programmatic quantization is still experimental and might break on certain input models with certain configurations.


For both these flows, the following options are exposed:
- Bit-width of weight and activations.
- Scales can be either float32 or power-of-two (po2) numbers.
- Weights' scale factors can be either per-tensor or per-channel.
- Biases can be int16 or int32.
- Activation quantization can be symmetric or asymmetric.
- Percentiles used for the activations' statistics computation during calibration.

Furthermore, Brevitas additional PTQ techniques can be enabled:
- Bias correction[<sup>1 </sup>].
- Graph equalization[<sup>1 </sup>].
- If Graph equalization is enabled, the _merge\_bias_ technique can be enabled.[<sup>2 </sup>] [<sup>3 </sup>].


Internally, when defining a quantized model programmatically, Brevitas leverages `torch.fx` and its `symbolic_trace` functionality, meaning that an input model is required to pass symbolic tracing for it to work.

For more information about what are the currently supported quantized layers in Brevitas, check the [following file](https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/quantize.py),
where we map the torch compute layers and activations with their corresponding quantized version.

## Evaluation flow

This flow allows to specify which pre-trained torchvision model to quantize and apply PTQ to with the desired quantization configuration.
It also gives the possibility to export the model to either ONNX QCDQ format or in torch QCDQ format.
The quantization and export options to specify are:
```bash
  -h, --help            show this help message and exit
  --calibration-dir CALIBRATION_DIR
                        Path to folder containing Imagenet calibration folder
  --validation-dir VALIDATION_DIR
                        Path to folder containing Imagenet validation folder
  --workers WORKERS     Number of data loading workers (default: 8)
  --batch-size-calibration BATCH_SIZE_CALIBRATION
                        Minibatch size for calibration (default: 64)
  --batch-size-validation BATCH_SIZE_VALIDATION
                        Minibatch size for validation (default: 256)
  --export-dir EXPORT_DIR
                        Directory where to store the exported models
  --gpu GPU             GPU id to use (default: None)
  --calibration-samples CALIBRATION_SAMPLES
                        Calibration size (default: 1000)
  --model-name ARCH     model architecture: alexnet | convnext_base |
                        convnext_large | convnext_small | convnext_tiny |
                        densenet121 | densenet161 | densenet169 | densenet201
                        | efficientnet_b0 | efficientnet_b1 | efficientnet_b2
                        | efficientnet_b3 | efficientnet_b4 | efficientnet_b5
                        efficientnet_v2_s | googlenet | inception_v3 |
                        list_models | maxvit_t | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 |
                        mobilenet_v3_large | mobilenet_v3_small |
                        regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf |
                        regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf |
                        regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf |
                        regnet_y_400mf | regnet_y_800mf | regnet_y_8gf |
                        resnet101 | resnet152 | resnet18 | resnet34 | resnet50
                        | resnext101_32x8d | resnext101_64x4d |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        swin_b | swin_s | swin_t | swin_v2_b | swin_v2_s |
                        swin_v2_t | vgg11 | vgg11_bn | vgg13 | vgg13_bn |
                        vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 |
                        vit_b_32 | vit_h_14 | vit_l_16 | vit_l_32 |
                        wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  --target-backend {generic,flexml}
                        Backend to target for quantization (default: generic)
  --scale-factor-type {float32,po2}
                        Type for scale factors (default: float32)
  --act-bit-width ACT_BIT_WIDTH
                        Activations bit width (default: 8)
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weights bit width (default: 8)
  --bias-bit-width {int32,int16}
                        Bias bit width (default: int32)
  --act-quant-type {symmetric,asymmetric}
                        Activation quantization type (default: symmetric)
  --graph-eq-iterations GRAPH_EQ_ITERATIONS
                        Numbers of iterations for graph equalization (default: 20)
  --act-quant-percentile ACT_QUANT_PERCENTILE
                        Percentile to use for stats of activation quantization (default: 99.999)
  --export-onnx-qcdq    If true, export the model in onnx qcdq format
  --export-torch-qcdq   If true, export the model in torch qcdq format
  --scaling-per-output-channel
                        Enable Weight scaling per output channel (default: enabled)
  --no-scaling-per-output-channel
                        Disable Weight scaling per output channel (default: enabled)
  --bias-corr           Enable Bias correction after calibration (default: enabled)
  --no-bias-corr        Disable Bias correction after calibration (default: enabled)
  --graph-eq-merge-bias
                        Enable Merge bias when performing graph equalization (default: enabled)
  --no-graph-eq-merge-bias
                        Disable Merge bias when performing graph equalization (default: enabled)
  --weight-narrow-range
                        Enable Narrow range for weight quantization (default: enabled)
  --no-weight-narrow-range
                        Disable Narrow range for weight quantization (default: enabled)
```

The script requires to specify the calibration folder (`--calibration-dir`), from which the calibration samples will be taken (configurable with the `--calibration-samples` argument), and a validation folder (`--validation-dir`).

## Benchmark flow

This scripts evaluate a variety of quantization configurations on different models.

For example, to run the script on the GPU 0:
```bash
brevitas_ptq_imagenet_benchmark --calibration-dir /path/to/imagenet/calibration/folder --validation-dir /path/to/imagenet/validation/folder --gpu 0
```

After launching the script, a `RESULT_TORCHVISION.md` markdown file will be generated with the results on the torchvision models,
and a `RESULTS_IMGCLSMOB.md` with the results on manually quantized models starting from floating point weights.

In this folder it is possible to find a pre-computed `RESULT_TORCHVISION.md` file all combinations evaluated on three different torchvision models (ResNet18, MobileNet V2, ViT B32),
as well as the results on the hand defined quantized MobileNet V1 (`RESULTS_IMGCLSMOB.md`).


[<sup>1 </sup>]: https://arxiv.org/abs/1906.04721
[<sup>2 </sup>]: https://github.com/Xilinx/Vitis-AI/blob/50da04ddae396d10a1545823aca30b3abb24a276/src/vai_quantizer/vai_q_pytorch/nndct_shared/optimization/commander.py#L450
[<sup>3 </sup>]: https://github.com/openppl-public/ppq/blob/master/ppq/quantization/algorithm/equalization.py
[<sup>4 </sup>]: https://github.com/osmr/imgclsmob
