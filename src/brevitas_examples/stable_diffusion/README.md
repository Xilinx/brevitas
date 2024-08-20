# Stable Diffusion Quantization

## Requirements

For MLPerf inference execution, it is recommended to follow the MLPerf instruction to download the dataset and all relevant files,
such as pre-generated latents and captions for calibration.

Similarly, a new python enviornment should be used with python<=3.10, installing first the requirements specified in
`requirements.txt` in stable_diffusion/mlperf_evaluation.


Afterwards, install brevitas with:
```bash
pip install -e .[export]
```

## Quantization Options

It supports Stable Diffusion 2.1 and Stable Diffusion XL.

The following PTQ techniques are currently supported:
- Activation Equalization (e.g., SmoothQuant), layerwise (with the addition of Mul ops)
- Activation Calibration, in the case of static activation quantization
- GPTQ
- Bias Correction

These techniques can be applied for both integer and floating point quantization.

Activation quantization is optional, and disabled by default. To enable, set both `conv-input-bit-width` and `linear-input-bit-width`.

We support ONNX integer export, and we are planning to release soon export for floating point quantization (e.g., FP8).

To export the model with fp16 scale factors, disable `export-cpu-float32`. This will performing the tracing necessary for export on GPU, leaving the model in fp16.
If the flag is not enabled, the model will be moved to CPU and cast to float32 before export because of missing CPU kernels in fp16.

To use MLPerf inference setup, check and install the correct requirements specified in the `requirements.txt` file under mlperf_evaluation.

For example, to perform weight-only quantization on SDXL, the following can be used:

`python main.py --resolution 1024 --batch 1 --model /path/to/sdxl --prompt 500 --conv-weight-bit-width 8 --linear-weight-bit-width 8 --dtype float16 --weight-quant-type sym  --calibration-steps 8 --guidance-scale 8. --use-negative-prompts --calibration-prompt 500  --activation-eq --use-mlperf`

To add activation quantization:

`--linear-input-bit 8 --conv-input-bit 8`

To choose between `static` or `dynamic` activation quantization, set the flag: `--input-scale-type` to either option

To include export:
`--export-target onnx`

To perform a dry-run quantization, where only the structure of the quantized model is preserved but no calibration of the quantized parameter is performed, add the `--dry-run` flag.



## Run

```bash
usage: main.py [-h] [-m MODEL] [-d DEVICE] [-b BATCH_SIZE] [--prompt PROMPT]
               [--calibration-prompt CALIBRATION_PROMPT]
               [--calibration-prompt-path CALIBRATION_PROMPT_PATH]
               [--checkpoint-name CHECKPOINT_NAME]
               [--load-checkpoint LOAD_CHECKPOINT]
               [--path-to-latents PATH_TO_LATENTS]
               [--path-to-coco PATH_TO_COCO] [--resolution RESOLUTION]
               [--guidance-scale GUIDANCE_SCALE]
               [--calibration-steps CALIBRATION_STEPS]
               [--output-path OUTPUT_PATH | --no-output-path]
               [--quantize | --no-quantize]
               [--activation-equalization | --no-activation-equalization]
               [--gptq | --no-gptq] [--bias-correction | --no-bias-correction]
               [--dtype {float32,float16,bfloat16}]
               [--attention-slicing | --no-attention-slicing]
               [--export-target {,onnx,params_only}]
               [--export-weight-q-node | --no-export-weight-q-node]
               [--conv-weight-bit-width CONV_WEIGHT_BIT_WIDTH]
               [--linear-weight-bit-width LINEAR_WEIGHT_BIT_WIDTH]
               [--conv-input-bit-width CONV_INPUT_BIT_WIDTH]
               [--act-eq-alpha ACT_EQ_ALPHA]
               [--linear-input-bit-width LINEAR_INPUT_BIT_WIDTH]
               [--linear-output-bit-width LINEAR_OUTPUT_BIT_WIDTH]
               [--weight-param-method {stats,mse}]
               [--input-param-method {stats,mse}]
               [--input-scale-stats-op {minmax,percentile}]
               [--input-zp-stats-op {minmax,percentile}]
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
               [--exclude-blacklist-act-eq | --no-exclude-blacklist-act-eq]
               [--quantize-input-zero-point | --no-quantize-input-zero-point]
               [--export-cpu-float32 | --no-export-cpu-float32]
               [--use-mlperf-inference | --no-use-mlperf-inference]
               [--use-negative-prompts | --no-use-negative-prompts]
               [--dry-run | --no-dry-run] [--quantize-sdp | --no-quantize-sdp]
               [--override-conv-quant-config | --no-override-conv-quant-config]
               [--vae-fp16-fix | --no-vae-fp16-fix]
               [--share-qkv-quant | --no-share-qkv-quant]

Stable Diffusion quantization

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path or name of the model.
  -d DEVICE, --device DEVICE
                        Target device for quantized model.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        How many seeds to use for each image during
                        validation. Default: 2
  --prompt PROMPT       Number of prompt to use for testing. Default: 4
  --calibration-prompt CALIBRATION_PROMPT
                        Number of prompt to use for calibration. Default: 2
  --calibration-prompt-path CALIBRATION_PROMPT_PATH
                        Path to calibration prompt
  --checkpoint-name CHECKPOINT_NAME
                        Name to use to store the checkpoint in the output dir.
                        If not provided, no checkpoint is saved.
  --load-checkpoint LOAD_CHECKPOINT
                        Path to checkpoint to load. If provided, PTQ
                        techniques are skipped.
  --path-to-latents PATH_TO_LATENTS
                        Load pre-defined latents. If not provided, they are
                        generated based on an internal seed.
  --path-to-coco PATH_TO_COCO
                        Path to MLPerf compliant Coco dataset. Used when the
                        --use-mlperf flag is set. Default: None
  --resolution RESOLUTION
                        Resolution along height and width dimension. Default:
                        512.
  --guidance-scale GUIDANCE_SCALE
                        Guidance scale.
  --calibration-steps CALIBRATION_STEPS
                        Steps used during calibration
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
  --bias-correction     Enable Toggle bias-correction. Default: Enabled
  --no-bias-correction  Disable Toggle bias-correction. Default: Enabled
  --dtype {float32,float16,bfloat16}
                        Model Dtype, choices are float32, float16, bfloat16.
                        Default: float16
  --attention-slicing   Enable Enable attention slicing. Default: Disabled
  --no-attention-slicing
                        Disable Enable attention slicing. Default: Disabled
  --export-target {,onnx,params_only}
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
                        Input bit width. Default: 0 (not quantized)
  --act-eq-alpha ACT_EQ_ALPHA
                        Alpha for activation equalization. Default: 0.9
  --linear-input-bit-width LINEAR_INPUT_BIT_WIDTH
                        Input bit width. Default: 0 (not quantized).
  --linear-output-bit-width LINEAR_OUTPUT_BIT_WIDTH
                        Input bit width. Default: 0 (not quantized).
  --weight-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --input-scale-stats-op {minmax,percentile}
                        Define what statics op to use for input scale.
                        Default: minmax.
  --input-zp-stats-op {minmax,percentile}
                        Define what statics op to use for input zero point.
                        Default: minmax.
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
                        X+Y==weight_bit_width-1. It's possible to add
                        float_ocp_ or float_fnuz_ before the exponent/mantissa
                        bitwidth. Default: int.
  --input-quant-format INPUT_QUANT_FORMAT
                        Input quantization type. Either int or eXmY, with
                        X+Y==input_bit_width-1. It's possible to add
                        float_ocp_ or float_fnuz_ before the exponent/mantissa
                        bitwidth. Default: int.
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
  --exclude-blacklist-act-eq
                        Enable Exclude unquantized layers from activation
                        equalization. Default: Disabled
  --no-exclude-blacklist-act-eq
                        Disable Exclude unquantized layers from activation
                        equalization. Default: Disabled
  --quantize-input-zero-point
                        Enable Quantize input zero-point. Default: Enabled
  --no-quantize-input-zero-point
                        Disable Quantize input zero-point. Default: Enabled
  --export-cpu-float32  Enable Export FP32 on CPU. Default: Disabled
  --no-export-cpu-float32
                        Disable Export FP32 on CPU. Default: Disabled
  --use-mlperf-inference
                        Enable Evaluate FID score with MLPerf pipeline.
                        Default: False
  --no-use-mlperf-inference
                        Disable Evaluate FID score with MLPerf pipeline.
                        Default: False
  --use-negative-prompts
                        Enable Use negative prompts during
                        generation/calibration. Default: Enabled
  --no-use-negative-prompts
                        Disable Use negative prompts during
                        generation/calibration. Default: Enabled
  --dry-run             Enable Generate a quantized model without any
                        calibration. Default: Disabled
  --no-dry-run          Disable Generate a quantized model without any
                        calibration. Default: Disabled
  --quantize-sdp        Enable Quantize SDP. Default: Disabled
  --no-quantize-sdp     Disable Quantize SDP. Default: Disabled
  --override-conv-quant-config
                        Enable Quantize Convolutions in the same way as SDP
                        (i.e., FP8). Default: Disabled
  --no-override-conv-quant-config
                        Disable Quantize Convolutions in the same way as SDP
                        (i.e., FP8). Default: Disabled
  --vae-fp16-fix        Enable Rescale the VAE to not go NaN with FP16.
                        Default: Disabled
  --no-vae-fp16-fix     Disable Rescale the VAE to not go NaN with FP16.
                        Default: Disabled
  --share-qkv-quant     Enable Share QKV/KV quantization. Default: Disabled
  --no-share-qkv-quant  Disable Share QKV/KV quantization. Default: Disabled

```
