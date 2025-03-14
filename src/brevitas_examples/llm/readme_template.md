# LLM quantization

## Requirements

- transformers
- datasets
- torch_mlir (optional for torch-mlir based export)
- optimum
- accelerate

## Run

Set the env variable `BREVITAS_JIT=1` to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

When using `--optimize-rotations`, the rotation training procedure relies on the Trainer class (https://huggingface.co/docs/transformers/en/main_classes/trainer). Therefore, training can be further configured by passing arguments accepted by the dataclass TrainingArguments (https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments), e.g. `--learning_rate`, `--weight_decay`, `per_device_train_batch_size`.

```bash
{{ readme_help }}
```
