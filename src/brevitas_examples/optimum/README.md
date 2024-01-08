# Prototype Integration of Brevitas and HuggingFace Optimum

The example with shows a prototype implementation of Brevitas and several of HuggingFace's tools interoperating.
The example shows:
 - Instantiation of facebook's OPT model from HuggingFace transformers
 - The definition and instantiation of a parametrizable `BrevitasQuantizer` which extends `OptimumQuantizer`
 - Optional conversion of the OPT model to an FX representation, leveraging HuggingFace transformers' tracer
 - Prototype support of executing PTQ algorithms and validation, while leveraging CPU offload from HuggingFace accelerate
 - Quantization of the OPT model using Brevitas' PTQ algorithms
   - Optionally converting the `OPTAttention` layer to `torch.nn.MultiheadAttention` for finer-grained quantization of MHA layers
   - Optionally applying: SmoothQuant, GPTQ, weight equalization algorithms
 - Validation of the quantized model
 - (WIP) Export to quantized ONNX (QDQ-style), leveraging HuggingFace optimum's ONNX export

## Prerequisites

The examples were tested using:
 - PyTorch v2.1.2
 - transformers v4.36.2
 - accelerate v0.25.0
 - optimum v1.16.1
 - brevitas (this branch)

## Running the Example

Set `HF_HOME` to your preferred path, then run:

```bash
python example.py --apply-act-equalization fx --with-fx
```

To quantize OPT with the graph-based SmoothQuant PTQ algorithm enabled.
For other options, run:

```bash
python example.py --help
```

Most options can be applied independently.
