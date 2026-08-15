# GGUF

Use Brevitas to calibrate and export quantized LLMs to [GGUF](https://github.com/ggml-org/ggml).
Run the exported models with llama.cpp, Ollama, and compatible runtimes.

## Quick Start

This quick start uses the following versions.

- `torch==2.6.0+rocm6.1`
- `gguf==0.18.0`
- `transformers==4.57.6`

See
[`requirements-llm.txt`](../../../../requirements/requirements-llm.txt) for the latest supported
versions.


## Quantization Recipes

Brevitas supports two types of GGUF quantization recipes.

### (Mostly) Uniform Recipes

Uniform recipes are registered in
[`custom_quantizers.py`](../../llm/gguf_export/custom_quantizers.py). Every linear layer uses the
same base quant type. When `quantize_first_last_layer: true`, high-impact tensors such as
`token_embd` and `output` use Q6_K. The Q8_0 and Q6_K recipes keep their base quant types.

| Name | Base weight quant | Notes |
|---|---|---|
| `gguf_q8_0` | Q8_0 | Uniform |
| `gguf_q6_k` | Q6_K | Uniform |
| `gguf_q5_k` | Q5_K | `token_embd` and `output` use Q6_K. This matches the `Q5_K_S` recipe from `llama-quantize`. |
| `gguf_q4_k` | Q4_K | `token_embd` and `output` use Q6_K. |
| `gguf_q4_1` | Q4_1 | `token_embd` and `output` use Q6_K. |
| `gguf_q4_0` | Q4_0 | `token_embd` and `output` use Q6_K. |
| `gguf_q3_k` | Q3_K | `token_embd` and `output` use Q6_K. This matches the `Q3_K_S` recipe from `llama-quantize`. |
| `gguf_q2_k` | Q2_K | `token_embd` and `output` use Q6_K. |

Use `--custom-quantizer=gguf_q8_0` on the command line. In YAML, use
`custom_quantizer: gguf_q8_0`.

### Mixed-Precision Recipe Plugins

The mixed-precision recipes mirror llama.cpp's `llama_tensor_get_type_impl` rules for the selected
models. Per-layer overrides use higher K-quants for sensitive tensors, such as
`self_attn.v_proj` and `mlp.down_proj`. When `quantize_first_last_layer: true`, `token_embd` and
`output` use Q6_K.

Specify a plugin as `path/to/recipe.py:quantizer_name`.

```yaml
custom_quantizer: recipes/Llama-3.2-1B.py:gguf_q4_k_m
```

| Plugin | Model | Registered names |
|---|---|---|
| [`recipes/Llama-3.2-1B.py`](recipes/Llama-3.2-1B.py) | Llama 3.2 1B (Base or Instruct) | `gguf_q4_0`, `gguf_q4_k_s`, `gguf_q4_k_m`, `gguf_q5_k_m`, `gguf_q2_k`, `gguf_q3_k_m`, `gguf_q3_k_l` |
| [`recipes/Llama-3.2-3B.py`](recipes/Llama-3.2-3B.py) | Llama 3.2 3B (Base or Instruct) | The same names as 1B. Layer rules differ. |

Recipe plugins validate the loaded model name via [`RecipeMixin`](recipes/common.py).

## Config Reference

Example configs set only non-default fields. See
[`default_template.yaml`](../../llm/config/default_template.yaml) for the full schema.

| Field | Purpose |
|---|---|
| `custom_quantizer` | A registered name, such as `gguf_q8_0`, or a plugin path, such as `recipes/Llama-3.2-1B.py:gguf_q4_k_s`. |
| `quantize_first_last_layer` | Required to also quantize `token_embd` and `output`. Without this option, these tensors stay F32 during export. |
| `export_target` | A `gguf:<ftype>` value, such as `gguf:q4_k_s` or `gguf:q8_0`. |
| `export_path` | The output file or directory. A path that ends in `.gguf` sets the exact output file. Other paths set an output directory. If omitted, the exporter writes to the working directory. |

> [!IMPORTANT]
> Run configs from this directory so relative `custom_quantizer` plugin paths resolve correctly.
> You can also use an absolute plugin path.

```bash
cd src/brevitas_examples/papers/gguf
brevitas_ptq_llm --config llama3-1b-q2_k.yml
```

### Example: mixed-precision recipe (Q4_K_S)

```yaml
model: meta-llama/Llama-3.2-1B-Instruct
custom_quantizer: recipes/Llama-3.2-1B.py:gguf_q4_k_s
dtype: float16
export_target: gguf:q4_k_s
export_path: Llama-3.2-1B-Instruct-Q4_K_S.gguf
quantize_first_last_layer: true
```

### Example: uniform quantizer (Q8_0)

```yaml
model: meta-llama/Llama-3.2-1B-Instruct
custom_quantizer: gguf_q8_0
dtype: float16
export_target: gguf:q8_0
export_path: Llama-3.2-1B-Instruct-Q8_0.gguf
quantize_first_last_layer: true
```

## Validation

The tables below compare the in-process Brevitas PPL with the canonical `llama-perplexity` PPL for each
exported GGUF file. Close agreement indicates that Brevitas emulation and GGUF export preserve the
expected llama.cpp model behavior.

For the reported results, we generated the corpus from the same WikiText2 text stream used by
[Brevitas evaluation](../../llm/llm_quant/eval.py). The join operation is defined in
[`data.py`](../../llm/llm_quant/data.py). The following example reproduces the corpus file:

```python
from pathlib import Path

from datasets import load_dataset

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(raw_dataset["text"])
Path("wiki.test.raw").write_text(text + "\n", encoding="utf-8")
```

The joined dataset already ends with a newline. We added one extra newline because
`llama-perplexity` removes exactly one trailing newline when it reads the file. After this removal,
the llama.cpp input matches the text stream that Brevitas uses. We evaluated each GGUF file with:

```bash
llama-perplexity -m <model.gguf> -f wiki.test.raw -c 2048 -ngl 0
```

This command processes all 141 chunks. Size values are reported in MB.

> [!NOTE]
> Small PPL differences are expected because the evaluation harnesses are not identical. For
> example, `llama-perplexity` replaces the first token in each `n`-token sequence with the BOS
> token. With `bos_preprocessing: sequence`, Brevitas prepends the BOS token to an `(n-1)`-token
> sequence. Other differences, such as backend kernels and emulated versus real quantization, can
> also affect PPL. The results remain close to the canonical evaluation. The largest measured gap
> is 0.98 PPL for the 1B Q2_K recipe.

### Llama-3.2-1B-Instruct

| Recipe | Size (MB) | Brevitas PPL | llama.cpp PPL |
|---|---:|---:|---:|
| Q8_0 | 1321.1 | 11.81 | 11.81 |
| Q6_K | 1021.8 | 11.82 | 11.82 |
| Q5_K_M | 911.5 | 11.90 | 11.89 |
| Q5_K | 892.6 | 11.99 | 12.01 |
| Q4_1 | 831.7 | 13.15 | 13.14 |
| Q4_K_M | 807.7 | 12.39 | 12.36 |
| Q4_K_S | 775.6 | 12.59 | 12.60 |
| Q4_0 | 773.0 | 13.81 | 13.77 |
| Q4_K | 770.9 | 13.04 | 13.02 |
| Q3_K_L | 732.5 | 13.34 | 13.36 |
| Q3_K_M | 690.8 | 14.16 | 14.12 |
| Q3_K_S | 641.7 | 18.22 | 18.10 |
| Q2_K | 580.9 | 31.58 | 30.60 |

### Llama-3.2-3B-Instruct

| Recipe | Size (MB) | Brevitas PPL | llama.cpp PPL |
|---|---:|---:|---:|
| Q8_0 | 3421.9 | 9.07 | 9.05 |
| Q6_K | 2643.9 | 9.06 | 9.03 |
| Q5_K_M | 2322.2 | 9.10 | 9.06 |
| Q5_K | 2269.5 | 9.18 | 9.14 |
| Q4_1 | 2093.4 | 9.36 | 9.36 |
| Q4_K_M | 2019.4 | 9.24 | 9.22 |
| Q4_K_S | 1928.2 | 9.37 | 9.36 |
| Q4_0 | 1921.9 | 9.75 | 9.70 |
| Q4_K | 1917.2 | 9.51 | 9.48 |
| Q3_K_L | 1815.3 | 9.77 | 9.69 |
| Q3_K_M | 1687.2 | 10.01 | 9.95 |
| Q3_K_S | 1542.8 | 11.59 | 11.44 |
| Q2_K | 1363.9 | 15.19 | 15.04 |
