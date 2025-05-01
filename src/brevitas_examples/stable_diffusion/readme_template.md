# Diffusion Quantization

This entrypoint supports quantization of different diffusion models, from Stable Diffusion to Flux.
We are still experimenting with it, thus some functionalities might not work as intended.
Feel free to open an issue if you face any issue.

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


The following PTQ techniques are currently supported:
- Activation Equalization (e.g., SmoothQuant), layerwise (with the addition of Mul ops)
- Activation Calibration, in the case of static activation quantization
- GPTQ
- SVDQuant
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
{{ readme_help }}
```
