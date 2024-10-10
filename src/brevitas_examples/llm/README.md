# LLM quantization

## Requirements

- transformers
- datasets
- torch_mlir (optional for torch-mlir based export)
- optimum
- optimum-amd (install from main)
- accelerate

## Run

Set the env variable BREVITAS_JIT=1 to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

```bash
usage: main.py [-h] [--model MODEL] [--seed SEED] [--nsamples NSAMPLES]
               [--seqlen SEQLEN] [--eval] [--dataset {wikitext2,c4}]
               [--weight-bit-width WEIGHT_BIT_WIDTH]
               [--weight-param-method {stats,mse,hqo}]
               [--weight-scale-precision {float_scale,po2_scale}]
               [--weight-quant-type {sym,asym}]
               [--weight-quant-format WEIGHT_QUANT_FORMAT]
               [--weight-quant-granularity {per_channel,per_tensor,per_group}]
               [--weight-group-dim {1,0}]
               [--weight-group-size WEIGHT_GROUP_SIZE]
               [--quantize-weight-zero-point]
               [--input-bit-width INPUT_BIT_WIDTH]
               [--input-quant-format INPUT_QUANT_FORMAT]
               [--input-param-method {stats,mse}]
               [--input-scale-precision {float_scale,po2_scale}]
               [--input-scale-type {static,dynamic,no_scale}]
               [--input-quant-type {sym,asym}]
               [--input-quant-granularity {per_tensor,per_row,per_group}]
               [--input-group-size INPUT_GROUP_SIZE]
               [--quantize-input-zero-point] [--quantize-last-layer] [--gptq]
               [--act-calibration] [--bias-corr] [--ln-affine-merge]
               [--no-quantize] [--no-float16] [--replace-mha]
               [--weight-equalization]
               [--act-equalization {None,layerwise,fx}] [--load-awq LOAD_AWQ]
               [--export-target {None,onnx_qcdq,torch_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}]
               [--export-prefix EXPORT_PREFIX]
               [--checkpoint-name CHECKPOINT_NAME] [--fuse-sequences]

options:
  -h, --help            show this help message and exit
  --model MODEL         HF model name. Default: facebook/opt-125m.
  --seed SEED           Seed for sampling the calibration data. Default: 0.
  --nsamples NSAMPLES   Number of calibration data samples. Default: 128.
  --seqlen SEQLEN       Sequence length. Default: 2048.
  --eval                Eval model PPL on the chosen Dataset.
  --dataset {wikitext2,c4}
                        Dataset to use for quantization (default: wikitext2)
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --weight-param-method {stats,mse,hqo}
                        How scales/zero-point are determined. Default: stats.
  --weight-scale-precision {float_scale,po2_scale}
                        Whether scale is a float value or a po2. Default: po2.
  --weight-quant-type {sym,asym}
                        Weight quantization type. Default: asym.
  --weight-quant-format WEIGHT_QUANT_FORMAT
                        Weight quantization type. Either int or eXmY, with
                        X+Y==weight_bit_width-1. It's possible to add
                        float_ocp_ or float_fnuz_ before the exponent/mantissa
                        bitwidth. Default: int.
  --weight-quant-granularity {per_channel,per_tensor,per_group}
                        Granularity for scales/zero-point of weights. Default:
                        per_group.
  --weight-group-dim {1,0}
                        Override default group_dim for groupsize quantization.
                        Default: layer-dependant
  --weight-group-size WEIGHT_GROUP_SIZE
                        Group size for per_group weight quantization. Default:
                        128.
  --quantize-weight-zero-point
                        Quantize weight zero-point.
  --input-bit-width INPUT_BIT_WIDTH
                        Input bit width. Default: None (disables input
                        quantization).
  --input-quant-format INPUT_QUANT_FORMAT
                        Input quantization type. Either int or eXmY, with
                        X+Y==weight_bit_width-1. It's possible to add
                        float_ocp_ or float_fnuz_ before the exponent/mantissa
                        bitwidth. Default: int.
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats
                        (percentile for static, absmax or minmax for dynamic).
  --input-scale-precision {float_scale,po2_scale}
                        Whether input scale is a float value or a po2.
                        Default: float.
  --input-scale-type {static,dynamic,no_scale}
                        Whether input scale is a static value or a dynamic
                        value.
  --input-quant-type {sym,asym}
                        Input quantization type. Default: asym.
  --input-quant-granularity {per_tensor,per_row,per_group}
                        Granularity for scales/zero-point of inputs. Default:
                        per_tensor.
  --input-group-size INPUT_GROUP_SIZE
                        Group size for per_group input quantization. Default:
                        64.
  --quantize-input-zero-point
                        Quantize input zero-point.
  --quantize-last-layer
                        Quantize last nn.Linear layer.
  --gptq                Apply GPTQ.
  --act-calibration     Apply activation calibration.
  --bias-corr           Apply bias correction.
  --ln-affine-merge     Merge LN affine params.
  --no-quantize         Disable quantization.
  --no-float16          Disable float16 as base datatype and switch to
                        float32.
  --replace-mha         Replace HuggingFace Attention with a quantizable
                        version
  --weight-equalization
                        Apply weight equalization. Relevant to ReLU based
                        models (e.g. OPT).
  --act-equalization {None,layerwise,fx}
                        Apply activation equalization (SmoothQuant). Layerwise
                        introduces standalone mul nodes,while fx merges them
                        whenever possible into previous tensors, which is
                        possible on ReLU based models (e.g. OPT).
  --load-awq LOAD_AWQ   Load the awq search results.
  --export-target {None,onnx_qcdq,torch_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}
                        Model export.
  --export-prefix EXPORT_PREFIX
                        Path prefix to use for the various export flows. If
                        None, a path will be derived from the model name
                        (default: None)
  --checkpoint-name CHECKPOINT_NAME
                        Filename to save checkpoint. If `None`, no checkpoint
                        is saved (default: None)
  --fuse-sequences      Whether to merge the dataset sequences in case they
                        are shorter than the requested number of samples per
                        sequence. This is useful in case you would like to
                        quantize or evaluate on long sequences (default:
                        False).

```
