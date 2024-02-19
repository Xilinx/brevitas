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
usage: ptq_evaluate.py [-h] --calibration-dir CALIBRATION_DIR --validation-dir
                       VALIDATION_DIR [--workers WORKERS]
                       [--batch-size-calibration BATCH_SIZE_CALIBRATION]
                       [--batch-size-validation BATCH_SIZE_VALIDATION]
                       [--export-dir EXPORT_DIR] [--gpu GPU]
                       [--calibration-samples CALIBRATION_SAMPLES]
                       [--model-name ARCH] [--dtype {float,bfloat16}]
                       [--target-backend {fx,layerwise,flexml}]
                       [--scale-factor-type {float_scale,po2_scale}]
                       [--act-bit-width ACT_BIT_WIDTH]
                       [--weight-bit-width WEIGHT_BIT_WIDTH]
                       [--layerwise-first-last-bit-width LAYERWISE_FIRST_LAST_BIT_WIDTH]
                       [--bias-bit-width {32,16,None}]
                       [--act-quant-type {sym,asym}]
                       [--weight-quant-type {sym,asym}]
                       [--weight-quant-granularity {per_tensor,per_channel}]
                       [--weight-quant-calibration-type {stats,mse}]
                       [--act-equalization {fx,layerwise,None}]
                       [--act-quant-calibration-type {stats,mse}]
                       [--graph-eq-iterations GRAPH_EQ_ITERATIONS]
                       [--learned-round-iters LEARNED_ROUND_ITERS]
                       [--learned-round-lr LEARNED_ROUND_LR]
                       [--act-quant-percentile ACT_QUANT_PERCENTILE]
                       [--export-onnx-qcdq] [--export-torch-qcdq]
                       [--scaling-per-output-channel | --no-scaling-per-output-channel]
                       [--bias-corr | --no-bias-corr]
                       [--graph-eq-merge-bias | --no-graph-eq-merge-bias]
                       [--weight-narrow-range | --no-weight-narrow-range]
                       [--gpfq-p GPFQ_P] [--quant-format {int,float}]
                       [--layerwise-first-last-mantissa-bit-width LAYERWISE_FIRST_LAST_MANTISSA_BIT_WIDTH]
                       [--layerwise-first-last-exponent-bit-width LAYERWISE_FIRST_LAST_EXPONENT_BIT_WIDTH]
                       [--weight-mantissa-bit-width WEIGHT_MANTISSA_BIT_WIDTH]
                       [--weight-exponent-bit-width WEIGHT_EXPONENT_BIT_WIDTH]
                       [--act-mantissa-bit-width ACT_MANTISSA_BIT_WIDTH]
                       [--act-exponent-bit-width ACT_EXPONENT_BIT_WIDTH]
                       [--accumulator-bit-width ACCUMULATOR_BIT_WIDTH]
                       [--onnx-opset-version ONNX_OPSET_VERSION]
                       [--channel-splitting-ratio CHANNEL_SPLITTING_RATIO]
                       [--gptq | --no-gptq] [--gpfq | --no-gpfq]
                       [--gpfa2q | --no-gpfa2q]
                       [--gpxq-act-order | --no-gpxq-act-order]
                       [--learned-round | --no-learned-round]
                       [--calibrate-bn | --no-calibrate-bn]
                       [--channel-splitting-split-input | --no-channel-splitting-split-input]
                       [--merge-bn | --no-merge-bn]

PyTorch ImageNet PTQ Validation

options:
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
                        | efficientnet_b6 | efficientnet_b7 |
                        efficientnet_v2_l | efficientnet_v2_m |
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
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        swin_b | swin_s | swin_t | swin_v2_b | swin_v2_s |
                        swin_v2_t | vgg11 | vgg11_bn | vgg13 | vgg13_bn |
                        vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 |
                        vit_b_32 | vit_h_14 | vit_l_16 | vit_l_32 |
                        wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  --dtype {float,bfloat16}
                        Data type to use
  --target-backend {fx,layerwise,flexml}
                        Backend to target for quantization (default: fx)
  --scale-factor-type {float_scale,po2_scale}
                        Type for scale factors (default: float_scale)
  --act-bit-width ACT_BIT_WIDTH
                        Activations bit width (default: 8)
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weights bit width (default: 8)
  --layerwise-first-last-bit-width LAYERWISE_FIRST_LAST_BIT_WIDTH
                        Input and weights bit width for first and last layer
                        w/ layerwise backend (default: 8)
  --bias-bit-width {32,16,None}
                        Bias bit width (default: 32)
  --act-quant-type {sym,asym}
                        Activation quantization type (default: sym)
  --weight-quant-type {sym,asym}
                        Weight quantization type (default: sym)
  --weight-quant-granularity {per_tensor,per_channel}
                        Activation quantization type (default: per_tensor)
  --weight-quant-calibration-type {stats,mse}
                        Weight quantization calibration type (default: stats)
  --act-equalization {fx,layerwise,None}
                        Activation equalization type (default: None)
  --act-quant-calibration-type {stats,mse}
                        Activation quantization calibration type (default:
                        stats)
  --graph-eq-iterations GRAPH_EQ_ITERATIONS
                        Numbers of iterations for graph equalization (default:
                        20)
  --learned-round-iters LEARNED_ROUND_ITERS
                        Numbers of iterations for learned round for each layer
                        (default: 1000)
  --learned-round-lr LEARNED_ROUND_LR
                        Learning rate for learned round (default: 1e-3)
  --act-quant-percentile ACT_QUANT_PERCENTILE
                        Percentile to use for stats of activation quantization
                        (default: 99.999)
  --export-onnx-qcdq    If true, export the model in onnx qcdq format
  --export-torch-qcdq   If true, export the model in torch qcdq format
  --scaling-per-output-channel
                        Enable Weight scaling per output channel (default:
                        enabled)
  --no-scaling-per-output-channel
                        Disable Weight scaling per output channel (default:
                        enabled)
  --bias-corr           Enable Bias correction after calibration (default:
                        enabled)
  --no-bias-corr        Disable Bias correction after calibration (default:
                        enabled)
  --graph-eq-merge-bias
                        Enable Merge bias when performing graph equalization
                        (default: enabled)
  --no-graph-eq-merge-bias
                        Disable Merge bias when performing graph equalization
                        (default: enabled)
  --weight-narrow-range
                        Enable Narrow range for weight quantization (default:
                        disabled)
  --no-weight-narrow-range
                        Disable Narrow range for weight quantization (default:
                        disabled)
  --gpfq-p GPFQ_P       P parameter for GPFQ (default: 1.0)
  --quant-format {int,float}
                        Quantization format to use for weights and activations
                        (default: int)
  --layerwise-first-last-mantissa-bit-width LAYERWISE_FIRST_LAST_MANTISSA_BIT_WIDTH
                        Mantissa bit width used with float layerwise
                        quantization for first and last layer (default: 4)
  --layerwise-first-last-exponent-bit-width LAYERWISE_FIRST_LAST_EXPONENT_BIT_WIDTH
                        Exponent bit width used with float layerwise
                        quantization for first and last layer (default: 3)
  --weight-mantissa-bit-width WEIGHT_MANTISSA_BIT_WIDTH
                        Mantissa bit width used with float quantization for
                        weights (default: 4)
  --weight-exponent-bit-width WEIGHT_EXPONENT_BIT_WIDTH
                        Exponent bit width used with float quantization for
                        weights (default: 3)
  --act-mantissa-bit-width ACT_MANTISSA_BIT_WIDTH
                        Mantissa bit width used with float quantization for
                        activations (default: 4)
  --act-exponent-bit-width ACT_EXPONENT_BIT_WIDTH
                        Exponent bit width used with float quantization for
                        activations (default: 3)
  --accumulator-bit-width ACCUMULATOR_BIT_WIDTH
                        Accumulator Bit Width for GPFA2Q (default: None)
  --onnx-opset-version ONNX_OPSET_VERSION
                        ONNX opset version
  --channel-splitting-ratio CHANNEL_SPLITTING_RATIO
                        Split Ratio for Channel Splitting. When set to 0.0,
                        Channel Splitting will not be applied. (default: 0.0)
  --gptq                Enable GPTQ (default: disabled)
  --no-gptq             Disable GPTQ (default: disabled)
  --gpfq                Enable GPFQ (default: disabled)
  --no-gpfq             Disable GPFQ (default: disabled)
  --gpfa2q              Enable GPFA2Q (default: disabled)
  --no-gpfa2q           Disable GPFA2Q (default: disabled)
  --gpxq-act-order      Enable GPxQ Act order heuristic (default: disabled)
  --no-gpxq-act-order   Disable GPxQ Act order heuristic (default: disabled)
  --learned-round       Enable Learned round (default: disabled)
  --no-learned-round    Disable Learned round (default: disabled)
  --calibrate-bn        Enable Calibrate BN (default: disabled)
  --no-calibrate-bn     Disable Calibrate BN (default: disabled)
  --channel-splitting-split-input
                        Enable Input Channels Splitting for channel splitting
                        (default: disabled)
  --no-channel-splitting-split-input
                        Disable Input Channels Splitting for channel splitting
                        (default: disabled)
  --merge-bn            Enable Merge BN layers before quantizing the model
                        (default: enabled)
  --no-merge-bn         Disable Merge BN layers before quantizing the model
                        (default: enabled)

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
