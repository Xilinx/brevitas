# LLM quantization

## Requirements

- transformers (from source)
- datasets
- torch_mlir (optional for torch-mlir based export)
- optimum (from source)
- optimum-amd (WIP, install brevitas-compatibility branch)
- accelerate (from source)

## Run

Set the env variable BREVITAS_JIT=1 to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

```bash
usage: main.py [-h] [--model MODEL] [--seed SEED] [--nsamples NSAMPLES] [--seqlen SEQLEN] [--eval] [--weight-bit-width WEIGHT_BIT_WIDTH] [--weight-param-method {stats,mse}]
               [--weight-scale-type {float32,po2}] [--weight-quant-type {sym,asym}] [--weight-quant-granularity {per_channel,per_tensor,per_group}]
               [--weight-group-size WEIGHT_GROUP_SIZE] [--quantize-weight-zero-point] [--input-bit-width INPUT_BIT_WIDTH] [--input-param-method {stats,mse}]
               [--input-scale-type {float32,po2}] [--input-quant-type {sym,asym}] [--input-quant-granularity {per_tensor}] [--quantize-input-zero-point] [--gptq]
               [--act-calibration] [--bias-corr] [--act-equalization]
               [--export-target {None,onnx_qcdq,torch_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         HF model name. Default: facebook/opt-125m.
  --seed SEED           Seed for sampling the calibration data. Default: 0.
  --nsamples NSAMPLES   Number of calibration data samples. Default: 128.
  --seqlen SEQLEN       Sequence length. Default: 2048.
  --eval                Eval model PPL on C4.
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --weight-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --weight-scale-type {float32,po2}
                        Whether scale is a float value or a po2. Default: po2.
  --weight-quant-type {sym,asym}
                        Weight quantization type. Default: asym.
  --weight-quant-granularity {per_channel,per_tensor,per_group}
                        Granularity for scales/zero-point of weights. Default: per_group.
  --weight-group-size WEIGHT_GROUP_SIZE
                        Group size for per_group weight quantization. Default: 128.
  --quantize-weight-zero-point
                        Quantize weight zero-point.
  --input-bit-width INPUT_BIT_WIDTH
                        Input bit width. Default: None (disables input quantization).
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats.
  --input-scale-type {float32,po2}
                        Whether input scale is a float value or a po2. Default: float32.
  --input-quant-type {sym,asym}
                        Input quantization type. Default: asym.
  --input-quant-granularity {per_tensor}
                        Granularity for scales/zero-point of inputs. Default: per_tensor.
  --quantize-input-zero-point
                        Quantize input zero-point.
  --gptq                Apply GPTQ.
  --act-calibration     Apply activation calibration.
  --bias-corr           Apply bias correction.
  --act-equalization    Apply activation equalization (SmoothQuant).
  --export-target {None,onnx_qcdq,torch_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}
                        Model export.
```
