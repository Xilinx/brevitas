# BF16 dMX Training Fix

## Summary

Full-size-rotation dMX training diverged in BF16 while FP32 worked at much
higher cost. The instability was caused by BF16 trainable state, not the
rotation matmul. Keeping the model and rotation forward in BF16 while storing
the learned bit-width and rotation parameters in FP32 recovers quality.

## Root Cause

Three issues compounded near the learned bit-width target:

1. Bit-width hooks collected FP32 values, but the aggregate regularization
   loss was cast back to the BF16 model dtype. The hinge penalty therefore
   changed in coarse steps near the target.
2. Learned bit-width offsets and their SGD momentum buffers were BF16.
3. CaileySGD computed rotations in FP32 but wrote the full-size rotation back
   to BF16 after every update, degrading orthogonality.

The fix keeps the aggregate loss in FP32 and adds FP32 master storage for
learned bit-width offsets and rotation matrices. The rotation is still cast to
BF16 at the existing forward boundary, so the full-size rotation matmul stays
BF16.

## Reproduction

Experiment setup:

- Model: `meta-llama/Llama-3.1-8B`
- `target_bit_width: 7.0`, `max_steps: 600`
- BF16 model, `compile_ptq: true`, batch size 4, gradient accumulation 2
- Full-size Hadamard rotations (`rotation_block_size: null`)
- `rotation_lr: 1.5`, `bw_learning_rate: 2.0`, `delay_start: 0.6`
- FineWeb training with wikitext2 validation fallback
- Float perplexity: 5.871

| Bit-width state | Rotation state | Rotation forward | Quant PPL |
| --- | --- | --- | ---: |
| BF16 | BF16 | BF16 | 50.552 |
| BF16 with FP32 aggregate loss and corrected scale floor | BF16 | BF16 | 30.972 |
| FP32 | BF16 | BF16 | 9.609 |
| FP32 | FP32 | BF16 | 6.037 |

The final run maintained sampled rotation orthogonality at about `4e-7`,
compared with roughly `1e-2` for BF16 rotation storage.

## Configuration

Enable the fix in a learned-float dMX configuration:

```yaml
rotation_parameter_dtype: float32
bit_width_parameter_dtype: float32
```

The `mxfp8_big_models_mixed_precision.yaml` and
`mxfp6_big_models_mixed_precision.yaml` benchmark configurations include both
keys for 3B-8B learned-float sweeps.

The learned quantizers now also receive the configured `scaling_min_val`
directly; this restores the requested `1e-4` floor instead of the prior
`1e-10` injector default.

## Environment

Use Python 3.11 with ROCm-compatible PyTorch and Triton, then install the
editable package and LLM dependencies. LightEval is needed when
`few_shot_eval: lighteval` is enabled.

```bash
uv venv --python 3.11 .venv
uv pip install <local-torch-wheel> <local-triton-wheel>
uv pip install -e . -r requirements/requirements-llm.txt \
  -r requirements/requirements-lighteval.txt pytest
```

Run the focused regression tests with:

```bash
.venv/bin/pytest -q tests/brevitas_examples/test_dmx_learned_float.py
```
