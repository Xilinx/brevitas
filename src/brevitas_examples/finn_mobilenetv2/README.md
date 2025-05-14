# Programmatic Quantization of MobileNetv2 for FINN

This tutorial / demo shows how to do programmatic quantization of a MobileNetv2 model for [FINN](https://github.com/xilinx/finn).
It includes applying several post-training quantization (PTQ) algoritms to the model, in order to maintain / recover model accuracy.
In future, this may be extended to include some quantization-aware training (QAT) if it is requested.

## Demo Overview

The demo shows 3 main aspects of Brevitas's quantization flow:
 - insertion of quantization nodes for inference-only compute;
 - maintaining / recovering accuracy with PTQ; and
 - export to [QONNX](https://github.com/fastmachinelearning/qonnx) for further processing with FINN.

## Environmental Setup

If [miniforge](https://github.com/conda-forge/miniforge) is installed, the environment can be set up as follows:

```bash
mamba env -n brevitas_finn_demo -f conda/brevitas_finn.yml
conda activate brevitas_finn_demo
pip install --no-deps /path/to/brevitas
```

## Running the Demo

By default, the demo quantizes the model weights and activation to 8 bits and can be run as following:

```bash
python finn_mobilenetv2.py
```

which should achieve approximately ~71.61% accuracy (from baseline of 72.01%).
Running the script produces a QONNX file `quant_mobilenet_v2.onnx`, which can be viewed with [Netron](https://github.com/lutzroeder/netron).
You should notice the following about the output model:
 - it contains Quant nodes for the weights for every Conv2d / Linear layer;
 - the activation before any Conv2d / Linear layer also has a Quant node;
 - Quant nodes before eltwise additions (or concatenations) have the same scale factors; and
 - Batchnorm layers are left in the network _without_ merging them into surrounding layers.

The above properties allow many models to be consumed the FINN's frontend.

## Modifying the Demo

There are several ways to modify the demo including:
 - changing bit-widths of weights / activations; and
 - recovering accuracy with PTQ algorithms.

### Changing the Bitwidth of Conv2d Layers

In Brevitas, there are multiple ways to achieve this.
You'll find a few suggested options commented in `finn_mobilenetv2.py`,
before the call to `quantize_finn()` a few different options.
For each modification, we strongly recommend viewing the resultant QONNX model in Netron to see how the models compute graph has changed.

We reproduce and explain the code snippets below:

#### Snippet 1

```python
finn_quant_maps = default_quantize_maps_finn()
finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = 4 # 1. Override Conv2d weights to have 4-bits
model = quantize_finn(model, **finn_quant_maps)
```

Adding / overriding the `nn.Conv2d` entry in the `compute_layer_map` argument to `quantize_finn` to set `weight_bit_width=4`,
as expected, sets the bit-width of the weights of all `Conv2d` layers to 4.

#### Snippet 2

```python
finn_quant_maps = default_quantize_maps_finn()
finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = lambda module: 8 if module.groups != 1 else 4 # 2. Groupwise Conv2ds @ 8-bits, the rest @ 4-bits
model = quantize_finn(model, **finn_quant_maps)
```

Alternatively, `weight_bit_width` and other parameters can be _lambda functions_ which can be a function of the module instance itself.
In this case, groupwise convolutions will remain at 8-bits, while all other convolutions will be at 4-bits.

#### Snippet 3

```python
finn_quant_maps = default_quantize_maps_finn()
finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = lambda module, name: 8 if module.groups != 1 or name == "features.0.0" else 4 # 3. Keep first conv in 8-bits otherwise same as above
model = quantize_finn(model, **finn_quant_maps)
```

If the signature of the lambda function has 2 arguments, its name will be passed along with the module itself.
This means that any function of the module or its name can be used to determine the quantization parameters.
In this case, the very first convolution (named `"features.0.0"` in PyTorch) also remains at 8-bits,
along with the groupwise convolutions, while the rest remain at 8-bits.

### Retaining and Recovering the Accuracy

You'll notice that reducing the weight bit-width of several layers in MobileNetv2 significantly reduces its accuracy.
However, most of this accuracy can be recovered by applying 1 or more PTQ algorithms to the model.
The demo has the following parameters to control which PTQ algorithms are applied to the model:

```python
act_eq = False # Apply act equalization
act_eq_alpha = 0.5 # [0.0 -> 1.0] Intuition: higher makes weights easier to quantize, lower makes the activations easier to quantize
act_eq_add_mul_node = False # Add extra elementwise mul nodes before activation quantization. If True, lower `alpha` seems to work better (`alpha=0.175`)
bias_corr = False # Apply bias correction
gptq = False # Apply GPTQ
gpfq = False # Apply GPFQ
```

These flags enable the application of:
 - [activation equalization](https://arxiv.org/abs/2211.10438);
 - [bias correction](https://arxiv.org/abs/1906.04721);
 - [GPTQ](https://arxiv.org/abs/2210.17323); and
 - [GPFQ](https://arxiv.org/abs/2201.11113).

We leave the explanation of these techniques to their respective papers,
but a good starting point is to set `act_eq=True`, `gpfq=True`, `bias_corr=True`.
Afterwhich, finding the combination of PTQ flags / settings if left to the user to maximise the accuracy.
If `act_eq_add_mul_node=True`, the compute graph will be augmented to include a channelwise multiplication before many activation quantization functions,
which may help to increase accuracy at the cost of passing that complexity to downstream tools (i.e., FINN).
GPTQ & GPFQ cannot likely be applied at the same time.
Brevitas has many more PTQ algorithms not included here, please see the [imagenet](../imagenet_classification) and [LLM](../llm) examples to see how they are applied.
