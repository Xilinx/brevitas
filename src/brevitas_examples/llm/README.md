# LLM quantization

## Requirements

- transformers
- datasets
- torch_mlir (optional for torch-mlir based export)
- optimum
- accelerate

## Run

Set the env variable BREVITAS_JIT=1 to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

```bash
usage: main.py [-h] [--model MODEL] [--seed SEED] [--nsamples NSAMPLES]
               [--seqlen SEQLEN] [--eval] [--dataset {wikitext2,c4}]
               [--gpxq-block-name GPXQ_BLOCK_NAME]
               [--weight-bit-width WEIGHT_BIT_WIDTH]
               [--weight-param-method {stats,mse,hqo}]
               [--weight-scale-precision {float_scale,po2_scale}]
               [--weight-quant-type {sym,asym}]
               [--weight-quant-format WEIGHT_QUANT_FORMAT]
               [--weight-quant-granularity {per_channel,per_tensor,per_group}]
               [--scale-rounding-func-type {round,ceil,floor}]
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
               [--learned-round-lr LEARNED_ROUND_LR]
               [--learned-round-scale-lr LEARNED_ROUND_SCALE_LR]
               [--learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM]
               [--learned-round-iters LEARNED_ROUND_ITERS]
               [--learned-round-scale] [--quantize-input-zero-point]
               [--quantize-last-layer] [--gptq] [--gpfq] [--gpxq-act-order]
               [--gpxq-use-quant-activations] [--gpxq-create-weight-orig]
               [--gpxq-max-accumulator-bit-width GPXQ_MAX_ACCUMULATOR_BIT_WIDTH]
               [--gpxq-max-accumulator-tile-size GPXQ_MAX_ACCUMULATOR_TILE_SIZE]
               [--act-calibration] [--bias-corr] [--ln-affine-merge]
               [--convert-layernorm-to-rmsnorm] [--replace-rmsnorm]
               [--no-quantize] [--no-float16]
               [--scaling-min-val SCALING_MIN_VAL] [--replace-mha]
               [--weight-equalization] [--rotation {fx,layerwise,fused_no_fx}]
               [--rotation-mode {had,ort}] [--rotation-orphan-sink]
               [--act-equalization {None,layerwise,fx}] [--load-awq LOAD_AWQ]
               [--export-target {None,onnx_qcdq,torch_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}]
               [--export-prefix EXPORT_PREFIX]
               [--checkpoint-name CHECKPOINT_NAME] [--fuse-sequences]
               [--learned-round {None,linear_round}]
               [--learned-round-fast-update]

options:
  -h, --help            show this help message and exit
  --model MODEL         HF model name. Default: facebook/opt-125m.
  --seed SEED           Seed for sampling the calibration data. Default: 0.
  --nsamples NSAMPLES   Number of calibration data samples. Default: 128.
  --seqlen SEQLEN       Sequence length. Default: 2048.
  --eval                Eval model PPL on the chosen Dataset.
  --dataset {wikitext2,c4}
                        Dataset to use for quantization (default: wikitext2)
  --gpxq-block-name GPXQ_BLOCK_NAME
                        Block name for faster GPxQ optimization. It works only
                        if FX is not needed (default: None)
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
  --scale-rounding-func-type {round,ceil,floor}
                        Rounding function to use with Po2 scale. Default:
                        None.
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
  --learned-round-lr LEARNED_ROUND_LR
                        Learning rate for learned round parameter
                        optimization. Default: 0.005
  --learned-round-scale-lr LEARNED_ROUND_SCALE_LR
                        Learning rate for scale optimization during round
                        learning. Default: 0.01
  --learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM
                        Learning rate for scale optimization during round
                        learning. Default: 0.9
  --learned-round-iters LEARNED_ROUND_ITERS
                        Number of iterations for learned round. Default: 200.
  --learned-round-scale
                        Learned scale factor together with round.
  --quantize-input-zero-point
                        Quantize input zero-point.
  --quantize-last-layer
                        Quantize last nn.Linear layer.
  --gptq                Apply GPTQ.
  --gpfq                Apply GPFQ.
  --gpxq-act-order      Apply GPxQ activation ordering.
  --gpxq-use-quant-activations
                        Use quantized activations in GPxQ.
  --gpxq-create-weight-orig
                        Create weight_orig in GPxQ.
  --gpxq-max-accumulator-bit-width GPXQ_MAX_ACCUMULATOR_BIT_WIDTH
                        Maximum accumulator bit width for GPxQ using AXE.
  --gpxq-max-accumulator-tile-size GPXQ_MAX_ACCUMULATOR_TILE_SIZE
                        Maximum accumulator tile size for GPxQ using AXE.
  --act-calibration     Apply activation calibration.
  --bias-corr           Apply bias correction.
  --ln-affine-merge     Merge LN affine params.
  --convert-layernorm-to-rmsnorm
                        Merge LN affine params.
  --replace-rmsnorm     Replace HF RMSNorms with Torch one.
  --no-quantize         Disable quantization.
  --no-float16          Disable float16 as base datatype and switch to
                        float32.
  --scaling-min-val SCALING_MIN_VAL
                        Minimum value to clamp scale to when using bf16 or
                        fp16 quantization.
  --replace-mha         Replace HuggingFace Attention with a quantizable
                        version
  --weight-equalization
                        Apply weight equalization. Relevant to ReLU based
                        models (e.g. OPT).
  --rotation {fx,layerwise,fused_no_fx}
                        Apply graph rotation equalization
  --rotation-mode {had,ort}
                        If GraphRotation is enabled, decide how to compute the
                        random rotation matrix that is fully fused. Online or
                        partial rotation will always be Hadamard
  --rotation-orphan-sink
                        If GraphRotation is enabled, decide wheter to add
                        standalone hadamard matrices for the unfused layers
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
  --learned-round {None,linear_round}
                        Whether to use learned round. If `None`, RTN is used
                        (default: None)
  --learned-round-fast-update
                        Whether to use fast update with learned round.
                        Prototype (default: False)

```
