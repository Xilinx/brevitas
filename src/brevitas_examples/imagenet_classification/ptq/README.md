# Post Training Quantization

This folder contains an example on how to use Brevitas PTQ flow to quantize torchvision models, as well as how to calibrate models that use Brevitas quantized modules.


We provide two workflows:
- A benchmark suite that will tests several quantization configurations on few selected models;
- An export script that, given a model name and a quantization configuration, evaluate its performance and allows to export it to ONNX format.

For both these flows, the following options are evaluated:
- Weights and Activations are quantized to 8 bit;
- Scale factors can be either Floating Point (FP) or Power of Two (Po2);
- Weights' scale factors can be either per-tensor or per-channel;
- Bias are quantized to int32 values for FP scale factors, int16 or int32 otherwise;
- Activations quantizer can be symmetric or asymmetric;
- Three different percentiles can be used for the activations' statistics computation (99.9, 99.99, 99.999).

Furthermore, Brevitas supports the following PTQ techniques:
- Bias Correction[<sup>1 </sup>];
- Graph Equalization[<sup>1 </sup>];
- If Graph Equalization is enabled, it  is possible to use the _merge\_bias_ technique.[<sup>2 </sup>] [<sup>3 </sup>].

It must be noted that Brevitas allows the user to define several other parameters, such as:
- The Bias Quantization type (Internal scaling vs External scaling);
- The bit width for every compute layer type (including the bias bit width) and for every activation type;
- The quantization configuration of specific layers (e.g., it is possible to define a different bit width only for the first and last compute layers);
- Several possible options for computing the scale factors of compute layers and activations.

For more information about what are the currently supported quantized layers in Brevitas, check the [following file](https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/quantize.py),
where we map the torch compute layers and activations with their corresponding quantized version.

## Benchmark flow

Starting from pretrained floating-point torchvision models, Brevitas offers the possibility to automatically obtain the corresponding quantized model leveraging torch.fx transformations.
For the selected subset of torchvision models, we test several possible combinations of the options described above.

The second type of benchmarks will run pre-defined quantized MobileNet v1, starting from the pre-trained FP weights[<sup>4 </sup>].
The pre-defined quantized model uses floating point scale factors, with a mix of per-tensor and per-channel strategies.
Weights and Activations are quantized at 8 bit.

To run the PTQ Benchmark suite on ImageNet simply make sure you have Brevitas installed and the ImageNet dataset in a Pytorch friendly format.

For example, to run the script on the GPU 0:
```bash
brevitas_ptq_imagenet_benchmark --calib-dir /path/to/imagenet/calibration/folder --validation-dir /path/to/imagenet/validation/folder --gpu 0
```
The script requires to specify the calibration folder (`--calib-dir`), from which the calibration samples will be taken (configurable with the `--calibration-samples` argument), and a validation folder (`--valid-dir`).

After launching the script, a `RESULT.md` markdown file will be generated two tables correspoding to the two types of benchmarks flows.


## Evaluation flow

This flow allows to manually specify a pre-trained torchvision model to quantize and calibrate with the desired quantization configuration.
It also gives the possibility to export the model in the ONNX QCDQ format or in torch QCDQ format.

The quantization and export options to specify are the following:
```bash
  -h, --help            show this help message and exit
  --calibration-dir CALIBRATION_DIR
                        path to folder containing Imagenet calibration folder
  --validation-dir VALIDATION_DIR
                        path to folder containing Imagenet validation folder
  --workers WORKERS     Number of data loading workers (default: 8)
  --batch-size-calibration BATCH_SIZE_CALIBRATION
                        Minibatch size for calibration (default: 64)
  --batch-size-validation BATCH_SIZE_VALIDATION
                        Minibatch size for validation (default: 256)
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
  --target-backend {generic,flexml}
                        Backend to target for quantization (default: generic)
  --scale-factor-type {float32,po2}
                        Type for scale factors (default: float32)
  --bit-width BIT_WIDTH
                        Weights and activations bit width (default: 8)
  --bias-bit-width {int32,int16}
                        Bias bit width (default: int32)
  --act-quant-type {symmetric,asymmetric}
                        Activation quantization type (default: symmetric)
  --graph-eq-iterations GRAPH_EQ_ITERATIONS
                        Numbers of iterations for graph equalization (default: 20)
  --act-quant-percentile ACT_QUANT_PERCENTILE
                        Percentile to use for stats of activation quantization
                        (default: 99.999)
  --export-path-onnx-qcdq EXPORT_PATH_ONNX_QCDQ
                        If specified, path where to export the model in onnx qcdq format
  --export-path-torch-qcdq EXPORT_PATH_TORCH_QCDQ
                        If specified, path where to export the model in torch
                        qcdq format (default: none)
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
```

The script requires to specify the calibration folder (`--calib-dir`), from which the calibration samples will be taken (configurable with the `--calibration-samples` argument), and a validation folder (`--valid-dir`)

For example, to run the script on the GPU 0:
```bash
brevitas_ptq_imagenet_evaluate --imagenet-dir /path/to/imagenet --gpu 0 --model-name resnet18 --scale-type po2 --act-quant-type asymmetric --act-quant-percentile 99.999 --export-path-qcdq ./quantized_model.onnx
```

[<sup>1 </sup>]: https://arxiv.org/abs/1906.04721
[<sup>2 </sup>]: https://github.com/Xilinx/Vitis-AI/blob/50da04ddae396d10a1545823aca30b3abb24a276/src/vai_quantizer/vai_q_pytorch/nndct_shared/optimization/commander.py#L450
[<sup>3 </sup>]: https://github.com/openppl-public/ppq/blob/master/ppq/quantization/algorithm/equalization.py
[<sup>4 </sup>]: https://github.com/osmr/imgclsmob
