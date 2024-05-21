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

To use MLPerf inference setup, check and install the correct requirements specified in the `requirements.txt` file under mlperf_evaluation.

For example, to perform weight-only quantization on SDXL, the following can be used:

`python main.py --resolution 1024 --batch 1 --model /path/to/sdxl --prompt 500 --conv-weight-bit-width 8 --linear-weight-bit-width 8 --dtype float16 --weight-quant-type sym  --calibration-steps 8 --guidance-scale 8. --use-negative-prompts --calibration-prompt 500  --activation-eq --use-mlperf`

To add activation quantization:

`--linear-input-bit 8 --conv-input-bit 8`

To choose between `static` or `dynamic` activation quantization, set the flag: `--input-scale-type` to either option

To include export:
`--export-target torch` or `--export-target onnx`

To perform a dry-run quantization, where only the structure of the quantized model is preserved but no calibration of the quantized parameter is performed, add the `--dry-run` flag.



## Run

```bash
usage: main.py [-h] [-m MODEL] [-d DEVICE] [-b BATCH_SIZE] [--prompt PROMPT]
               [--calibration-prompt CALIBRATION_PROMPT]
               [--calibration-prompt-path CALIBRATION_PROMPT_PATH]
               [--checkpoint-name CHECKPOINT_NAME]
               [--load-checkpoint LOAD_CHECKPOINT]
               [--path-to-latents PATH_TO_LATENTS] [--resolution RESOLUTION]
               [--guidance-scale GUIDANCE_SCALE]
               [--calibration-steps CALIBRATION_STEPS]
               [--output-path OUTPUT_PATH | --no-output-path]
               [--quantize | --no-quantize]
               [--activation-equalization | --no-activation-equalization]
               [--gptq | --no-gptq] [--bias-correction | --no-bias-correction]
               [--dtype {float32,float16,bfloat16}]
               [--attention-slicing | --no-attention-slicing]
               [--export-target {,torch,onnx}]
               [--export-weight-q-node | --no-export-weight-q-node]
               [--conv-weight-bit-width CONV_WEIGHT_BIT_WIDTH]
               [--linear-weight-bit-width LINEAR_WEIGHT_BIT_WIDTH]
               [--conv-input-bit-width CONV_INPUT_BIT_WIDTH]
               [--linear-input-bit-width LINEAR_INPUT_BIT_WIDTH]
               [--weight-param-method {stats,mse}]
               [--input-param-method {stats,mse}]
               [--input-stats-op {minmax,percentile}]
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
               [--use-mlperf-inference | --no-use-mlperf-inference]
               [--use-ocp | --no-use-ocp]
               [--use-negative-prompts | --no-use-negative-prompts]
               [--dry-run | --no-dry-run]

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
  --prompt PROMPT       Number of prompt to use for testing. Default: 4. Max:
                        4
  --calibration-prompt CALIBRATION_PROMPT
                        Number of prompt to use for calibration. Default: 2
  --calibration-prompt-path CALIBRATION_PROMPT_PATH
                        Path to calibration prompt
  --checkpoint-name CHECKPOINT_NAME
                        Name to use to store the checkpoint. If not provided,
                        no checkpoint is saved.
  --load-checkpoint LOAD_CHECKPOINT
                        Path to checkpoint to load. If provided, PTQ
                        techniques are skipped.
  --path-to-latents PATH_TO_LATENTS
                        Load pre-defined latents. If not provided, they are
                        generated based on an internal seed.
  --resolution RESOLUTION
                        Resolution along height and width dimension. Default:
                        512.
  --guidance-scale GUIDANCE_SCALE
                        Guidance scale.
  --calibration-steps CALIBRATION_STEPS
                        Percentage of steps used during calibration
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
  --export-target {,torch,onnx}
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
  --input-stats-op {minmax,percentile}
                        Define what statics op to use . Default: minmax.
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
  --use-mlperf-inference
                        Enable Evaluate FID score with MLPerf pipeline.
                        Default: False
  --no-use-mlperf-inference
                        Disable Evaluate FID score with MLPerf pipeline.
                        Default: False
  --use-ocp             Enable Use OCP format for float quantization. Default:
                        True
  --no-use-ocp          Disable Use OCP format for float quantization.
                        Default: True
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

```
