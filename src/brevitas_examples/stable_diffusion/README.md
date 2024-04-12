# Stable Diffusion Quantization

It supports Stable Diffusion 2.1 and Stable Diffusion XL.

The following PTQ techniques are currently supported:
- Activation Equalization (e.g., SmoothQuant), layerwise (with the addition of Mul ops)
- Activation Calibration, in the case of static activation quantization
- GPTQ
- Bias Correction

These techniques can be applied for both integer and floating point quantization.
Activation quantization is optional, and disabled by default. To enable, set both `conv-input-bit-width` and `linear-input-bit-width`.

We support ONNX integer export, and we are planning to release soon export for floating point quantization (e.g., FP8).

To export the model with fp16 scale factors, enable `export-cuda-float16`. This will performing the tracing necessary for export on GPU, leaving the model in fp16.
If the flag is not enabled, the model will be moved to CPU and cast to float32 before export because of missing CPU kernels in fp16.


## Run

```bash
usage: main.py [-h] [-m MODEL] [-d DEVICE] [-b BATCH_SIZE] [--prompt PROMPT]
               [--resolution RESOLUTION]
               [--output-path OUTPUT_PATH | --no-output-path]
               [--quantize | --no-quantize]
               [--activation-equalization | --no-activation-equalization]
               [--gptq | --no-gptq] [--float16 | --no-float16]
               [--attention-slicing | --no-attention-slicing]
               [--export-target {,onnx}]
               [--export-weight-q-node | --no-export-weight-q-node]
               [--conv-weight-bit-width CONV_WEIGHT_BIT_WIDTH]
               [--linear-weight-bit-width LINEAR_WEIGHT_BIT_WIDTH]
               [--conv-input-bit-width CONV_INPUT_BIT_WIDTH]
               [--linear-input-bit-width LINEAR_INPUT_BIT_WIDTH]
               [--weight-param-method {stats,mse}]
               [--input-param-method {stats,mse}]
               [--weight-scale-precision {float_scale,po2_scale}]
               [--input-scale-precision {float_scale,po2_scale}]
               [--weight-quant-type {sym,asym}]
               [--input-quant-type {sym,asym}]
               [--weight-quant-format WEIGHT_QUANT_FORMAT]
               [--input-quant-format INPUT_QUANT_FORMAT]
               [--weight-quant-granularity {per_channel,per_tensor,per_group}]
               [--input-quant-granularity {per_tensor}]
               [--input-scale-type {static,dynamic}]
               [--weight-group-size WEIGHT_GROUP_SIZE]
               [--quantize-weight-zero-point | --no-quantize-weight-zero-point]
               [--export-cuda-float16 | --no-export-cuda-float16]

Stable Diffusion quantization

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path or name of the model.
  -d DEVICE, --device DEVICE
                        Target device for quantized model.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Default: 4
  --prompt PROMPT       Manual prompt for testing. Default: An austronaut
                        riding a horse on Mars.
  --resolution RESOLUTION
                        Resolution along height and width dimension. Default:
                        512.
  --output-path OUTPUT_PATH
                        Path where to generate output folder.
  --no-output-path      Disable Path where to generate output folder.
  --quantize            Enable Toggle quantization. Default: Enabled
  --no-quantize         Disable Toggle quantization. Default: Enabled
  --activation-equalization
                        Enable Toggle Activation Equalization. Default:
                        Disabled
  --no-activation-equalization
                        Disable Toggle Activation Equalization. Default:
                        Disabled
  --gptq                Enable Toggle gptq. Default: Disabled
  --no-gptq             Disable Toggle gptq. Default: Disabled
  --float16             Enable Enable float16 execution. Default: Enabled
  --no-float16          Disable Enable float16 execution. Default: Enabled
  --attention-slicing   Enable Enable attention slicing. Default: Disabled
  --no-attention-slicing
                        Disable Enable attention slicing. Default: Disabled
  --export-target {,onnx}
                        Target export flow.
  --export-weight-q-node
                        Enable Enable export of floating point weights + QDQ
                        rather than integer weights + DQ. Default: Disabled
  --no-export-weight-q-node
                        Disable Enable export of floating point weights + QDQ
                        rather than integer weights + DQ. Default: Disabled
  --conv-weight-bit-width CONV_WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --linear-weight-bit-width LINEAR_WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --conv-input-bit-width CONV_INPUT_BIT_WIDTH
                        Input bit width. Default: None (not quantized)
  --linear-input-bit-width LINEAR_INPUT_BIT_WIDTH
                        Input bit width. Default: None (not quantized).
  --weight-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --weight-scale-precision {float_scale,po2_scale}
                        Whether scale is a float value or a po2. Default:
                        float_scale.
  --input-scale-precision {float_scale,po2_scale}
                        Whether scale is a float value or a po2. Default:
                        float_scale.
  --weight-quant-type {sym,asym}
                        Weight quantization type. Default: asym.
  --input-quant-type {sym,asym}
                        Input quantization type. Default: asym.
  --weight-quant-format WEIGHT_QUANT_FORMAT
                        Weight quantization type. Either int or eXmY, with
                        X+Y==weight_bit_width-1. Default: int.
  --input-quant-format INPUT_QUANT_FORMAT
                        Input quantization type. Either int or eXmY, with
                        X+Y==input_bit_width-1. Default: int.
  --weight-quant-granularity {per_channel,per_tensor,per_group}
                        Granularity for scales/zero-point of weights. Default:
                        per_channel.
  --input-quant-granularity {per_tensor}
                        Granularity for scales/zero-point of inputs. Default:
                        per_tensor.
  --input-scale-type {static,dynamic}
                        Whether to do static or dynamic input quantization.
                        Default: static.
  --weight-group-size WEIGHT_GROUP_SIZE
                        Group size for per_group weight quantization. Default:
                        16.
  --quantize-weight-zero-point
                        Enable Quantize weight zero-point. Default: Enabled
  --no-quantize-weight-zero-point
                        Disable Quantize weight zero-point. Default: Enabled
  --export-cuda-float16
                        Enable Export FP16 on CUDA. Default: Disabled
  --no-export-cuda-float16
                        Disable Export FP16 on CUDA. Default: Disabled
```
