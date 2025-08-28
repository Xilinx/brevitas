# LLM quantization

## Requirements

The requirements can be installed as followed:

```bash
pip install -e .[export,llm]
```

## Run

Set the env variable `BREVITAS_JIT=1` to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

When using `--optimize-rotations`, the rotation training procedure relies on the Trainer class (https://huggingface.co/docs/transformers/en/main_classes/trainer). Therefore, training can be further configured by passing arguments accepted by the dataclass TrainingArguments (https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments), e.g. `--learning_rate`, `--weight_decay`, `per_device_train_batch_size`.

```bash
usage: main.py [-h] [--config CONFIG] [--model MODEL] [--dtype {auto,float32,float16,bfloat16}] [--seed SEED] [--nsamples NSAMPLES] [--nsamples-rot-calibration NSAMPLES_ROT_CALIBRATION] [--seqlen SEQLEN]
               [--eval] [--dataset {wikitext2,c4,pile}] [--gpxq-block-name GPXQ_BLOCK_NAME] [--weight-bit-width WEIGHT_BIT_WIDTH] [--weight-param-method {stats,mse,hqo}]
               [--weight-scale-precision {float_scale,po2_scale}] [--weight-quant-type {sym,asym}] [--weight-quant-format WEIGHT_QUANT_FORMAT] [--weight-quant-granularity {per_channel,per_tensor,per_group}]
               [--scale-rounding-func-type {round,ceil,floor}] [--weight-group-dim {1,0}] [--weight-group-size WEIGHT_GROUP_SIZE] [--quantize-weight-zero-point] [--input-bit-width INPUT_BIT_WIDTH]
               [--input-quant-format INPUT_QUANT_FORMAT] [--input-param-method {stats,mse}] [--input-scale-precision {float_scale,po2_scale}] [--input-scale-type {static,dynamic,no_scale}]
               [--input-quant-type {sym,asym}] [--input-quant-granularity {per_tensor,per_row,per_group}] [--input-group-size INPUT_GROUP_SIZE] [--attn-quant-config {qkvs,kv}]
               [--attn-bit-width ATTN_BIT_WIDTH] [--attn-quant-format ATTN_QUANT_FORMAT] [--attn-param-method {stats,mse}] [--attn-scale-precision {float_scale,po2_scale}]
               [--attn-scale-type {static,dynamic,no_scale}] [--attn-quant-type {sym,asym}] [--attn-quant-granularity {per_tensor,per_row,per_group}] [--attn-group-size ATTN_GROUP_SIZE]
               [--learned-round-lr LEARNED_ROUND_LR] [--learned-round-scale-lr LEARNED_ROUND_SCALE_LR] [--learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM]
               [--learned-round-iters LEARNED_ROUND_ITERS] [--learned-round-scale] [--quantize-input-zero-point] [--quantize-last-layer] [--magr] [--magr-alpha MAGR_ALPHA] [--qronos]
               [--qronos-alpha QRONOS_ALPHA] [--gptq] [--gpfq] [--gpxq-act-order] [--gpxq-use-quant-activations] [--disable-create-weight-orig]
               [--gpxq-max-accumulator-bit-width GPXQ_MAX_ACCUMULATOR_BIT_WIDTH] [--gpxq-max-accumulator-tile-size GPXQ_MAX_ACCUMULATOR_TILE_SIZE] [--act-calibration] [--bias-corr] [--ln-affine-merge]
               [--convert-layernorm-to-rmsnorm] [--replace-rmsnorm] [--no-quantize] [--scaling-min-val SCALING_MIN_VAL] [--quant-sdpa {eager,functional,fx}] [--eager-quant-sdpa-class EAGER_QUANT_SDPA_CLASS]
               [--weight-equalization] [--rotation {fx,layerwise,fused_no_fx}] [--optimize-rotations] [--rotation-mode {had,ort}] [--rotation-orphan-sink] [--rotation-sdpa-regions]
               [--rotation-layers-to-expand [ROTATION_LAYERS_TO_EXPAND ...]] [--expansion-step EXPANSION_STEP] [--svd-quant] [--svd-quant-rank SVD_QUANT_RANK] [--svd-quant-iters SVD_QUANT_ITERS]
               [--act-equalization {None,layerwise,fx}] [--act-equalization-alpha ACT_EQUALIZATION_ALPHA]
               [--export-target {None,onnx_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}] [--export-prefix EXPORT_PREFIX] [--checkpoint-name CHECKPOINT_NAME] [--load-checkpoint]
               [--learned-round {None,linear_round}] [--learned-round-fast-update] [--few-shot-eval {lm_eval,lighteval}] [--few-shot-override-batch-size FEW_SHOT_OVERRIDE_BATCH_SIZE] [--compile-ptq]
               [--compile-eval] [--few-shot-zeroshot] [--bos-preprocessing {None,document,sequence}] [--few-shot-limit FEW_SHOT_LIMIT] [--few-shot-tasks [FEW_SHOT_TASKS ...]] [--awq-scale] [--awq-clip]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Specify alternative default commandline args (e.g., config/default_template.yml). Default: None.
  --model MODEL         HF model name. Default: facebook/opt-125m.
  --dtype {auto,float32,float16,bfloat16}
                        Data type for model. Default: auto
  --seed SEED           Seed for sampling the calibration data. Default: 0.
  --nsamples NSAMPLES   Number of calibration data samples. Default: 128.
  --nsamples-rot-calibration NSAMPLES_ROT_CALIBRATION
                        Number of calibration data samples for rotation. Default: 800.
  --seqlen SEQLEN       Sequence length. Default: 2048.
  --eval                Eval model PPL on the chosen Dataset.
  --dataset {wikitext2,c4,pile}
                        Dataset to use for quantization (default: wikitext2)
  --gpxq-block-name GPXQ_BLOCK_NAME
                        Block name for faster GPxQ optimization. It works only if FX is not needed (default: None)
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --weight-param-method {stats,mse,hqo}
                        How scales/zero-point are determined. Default: stats.
  --weight-scale-precision {float_scale,po2_scale}
                        Whether scale is a float value or a po2. Default: po2.
  --weight-quant-type {sym,asym}
                        Weight quantization type. Default: asym.
  --weight-quant-format WEIGHT_QUANT_FORMAT
                        Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.
  --weight-quant-granularity {per_channel,per_tensor,per_group}
                        Granularity for scales/zero-point of weights. Default: per_group.
  --scale-rounding-func-type {round,ceil,floor}
                        Rounding function to use with Po2 scale. Default: None.
  --weight-group-dim {1,0}
                        Override default group_dim for groupsize quantization. Default: layer-dependant
  --weight-group-size WEIGHT_GROUP_SIZE
                        Group size for per_group weight quantization. Default: 128.
  --quantize-weight-zero-point
                        Quantize weight zero-point.
  --input-bit-width INPUT_BIT_WIDTH
                        Input bit width. Default: None (disables input quantization).
  --input-quant-format INPUT_QUANT_FORMAT
                        Input quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats (percentile for static, absmax or minmax for dynamic).
  --input-scale-precision {float_scale,po2_scale}
                        Whether input scale is a float value or a po2. Default: float.
  --input-scale-type {static,dynamic,no_scale}
                        Whether input scale is a static value or a dynamic value.
  --input-quant-type {sym,asym}
                        Input quantization type. Default: asym.
  --input-quant-granularity {per_tensor,per_row,per_group}
                        Granularity for scales/zero-point of inputs. Default: per_tensor.
  --input-group-size INPUT_GROUP_SIZE
                        Group size for per_group input quantization. Default: 64.
  --attn-quant-config {qkvs,kv}
                        Decide which parts of attention should be quantized. "kv" will only quantize KV, "qkvs" will quantize all MatMuls in attention (QKV & Softmax output). Note: --quant-sdpa needs be set
                        for this to have an effect. Default: qkvs
  --attn-bit-width ATTN_BIT_WIDTH
                        Attention bit width. Default: None (same as input).
  --attn-quant-format ATTN_QUANT_FORMAT
                        Attention quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: None
                        (same as input).
  --attn-param-method {stats,mse}
                        How scales/zero-point are determined. Default: None (same as input).
  --attn-scale-precision {float_scale,po2_scale}
                        Whether input scale is a float value or a po2. Default: (same as input).
  --attn-scale-type {static,dynamic,no_scale}
                        Whether input scale is a static value or a dynamic value. Default: (same as input).
  --attn-quant-type {sym,asym}
                        Input quantization type. Default: (same as input).
  --attn-quant-granularity {per_tensor,per_row,per_group}
                        Granularity for scales/zero-point of inputs. Default: (same as input).
  --attn-group-size ATTN_GROUP_SIZE
                        Group size for per_group input quantization. Default: (same as input).
  --learned-round-lr LEARNED_ROUND_LR
                        Learning rate for learned round parameter optimization. Default: 0.005
  --learned-round-scale-lr LEARNED_ROUND_SCALE_LR
                        Learning rate for scale optimization during round learning. Default: 0.01
  --learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM
                        Learning rate for scale optimization during round learning. Default: 0.9
  --learned-round-iters LEARNED_ROUND_ITERS
                        Number of iterations for learned round. Default: 200.
  --learned-round-scale
                        Learned scale factor together with round.
  --quantize-input-zero-point
                        Quantize input zero-point.
  --quantize-last-layer
                        Quantize last nn.Linear layer.
  --magr                Apply MagR.
  --magr-alpha MAGR_ALPHA
                        Alpha for MagR. Default: 0.01.
  --qronos              Apply Qronos.
  --qronos-alpha QRONOS_ALPHA
                        Alpha for Qronos. Default: 1e-6
  --gptq                Apply GPTQ.
  --gpfq                Apply GPFQ.
  --gpxq-act-order      Apply GPxQ activation ordering.
  --gpxq-use-quant-activations
                        Use quantized activations in GPxQ.
  --disable-create-weight-orig
                        Disable maintaining original weights for non-quant forward pass. Default: false
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
  --scaling-min-val SCALING_MIN_VAL
                        Minimum value to clamp scale to when using bf16 or fp16 quantization.
  --quant-sdpa {eager,functional,fx}
                        Define how to quantize SDPA. (default: None)
  --eager-quant-sdpa-class EAGER_QUANT_SDPA_CLASS
                        If quant_sdpa is eager, specify the name of the attention class. (default: auto)
  --weight-equalization
                        Apply weight equalization. Relevant to ReLU based models (e.g. OPT).
  --rotation {fx,layerwise,fused_no_fx}
                        Apply graph rotation equalization
  --optimize-rotations  Whether to optimize the rotations (default: False).
  --rotation-mode {had,ort}
                        If GraphRotation is enabled, decide how to compute the random rotation matrix that is fully fused. Online or partial rotation will always be Hadamard
  --rotation-orphan-sink
                        If GraphRotation is enabled, decide wheter to add standalone hadamard matrices for the unfused layers
  --rotation-sdpa-regions
                        If GraphRotation is enabled, decide wheter to equalize across SDPA
  --rotation-layers-to-expand [ROTATION_LAYERS_TO_EXPAND ...]
                        A list of module names to expand with hadamard rotation. Default: []
  --expansion-step EXPANSION_STEP
                        When layer expansion is set, decide how much to increase the layer sizes. Default: 1
  --svd-quant           Apply SVDQuant.
  --svd-quant-rank SVD_QUANT_RANK
                        Rank to use for SVDQuant (default: 32).
  --svd-quant-iters SVD_QUANT_ITERS
                        Number of iterations to use for SVDQuant (default: 1).
  --act-equalization {None,layerwise,fx}
                        Apply activation equalization (SmoothQuant). Layerwise introduces standalone mul nodes,while fx merges them whenever possible into previous tensors, which is possible on ReLU based
                        models (e.g. OPT).
  --act-equalization-alpha ACT_EQUALIZATION_ALPHA
                        If activation equalization is enabled, decide what alpha to use
  --export-target {None,onnx_qcdq,sharded_torchmlir_group_weight,sharded_packed_torchmlir_group_weight}
                        Model export.
  --export-prefix EXPORT_PREFIX
                        Path prefix to use for the various export flows. If None, a path will be derived from the model name (default: None)
  --checkpoint-name CHECKPOINT_NAME
                        Filename to save checkpoint. If `None`, no checkpoint is saved (default: None)
  --load-checkpoint     Boolean flag to load_checkpoint, uses checkpoint_name. Default False)
  --learned-round {None,linear_round}
                        Whether to use learned round. If `None`, RTN is used (default: None)
  --learned-round-fast-update
                        Whether to use fast update with learned round. Prototype (default: False)
  --few-shot-eval {lm_eval,lighteval}
                        Perform zero_shot evaluation with lm_eval or lighteval. Default None)
  --few-shot-override-batch-size FEW_SHOT_OVERRIDE_BATCH_SIZE
  --compile-ptq         Compile for PTQ algorithms. Default False)
  --compile-eval        Compile during evaluation. Default False)
  --few-shot-zeroshot   Whether to do zero or few shot eval. Default False)
  --bos-preprocessing {None,document,sequence}
                        Type of BOS token pre-processing for training and evaluation datasets. Default None)
  --few-shot-limit FEW_SHOT_LIMIT
                        Few shot limit. Default None)
  --few-shot-tasks [FEW_SHOT_TASKS ...]
                        A list of tasks for zero_shot evaluation. Default: ['arc_challenge', 'arc_easy', 'winogrande', 'piqa']
  --awq-scale           Whether to apply AWQ scaling (default: False).
  --awq-clip            Whether to apply AWQ clipping (default: False).
  ```
