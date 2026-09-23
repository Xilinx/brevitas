====================
GGUF Export
====================


GGML and GGUF have established as popular libraries and format to quantize and export LLM,
with several libraries being able to read GGUF and apply optimized inference.

In its current status, GGUF provides lots of flexibility in terms of representation, with several 
quantization options, but it also has some limitations, such as:

* No graph representation during export
* Mostly focused on weight-only quantization
* Limited optimization possibilities during QAT and/or PTQ

In Brevitas, we are progressively adding better support for GGUF export.
Currently the supported formats are:

* Q8_0
* Q4_0
* Q4_1
* Q4_K

The first three modes can be obtained through our LLM entrypoint (`brevitas_ptq_llm`),
while it is possible to target all the formats through direct quantization.

The specification for these formats can be found `here <https://huggingface.co/docs/hub/gguf>`_.

LLM Entrypoint
==============

Brevitas' LLM entrypoint allows the user to load, quantize, test, and export many of the LLM available on 
HuggingFace, by simply passing a series of command line arguments that can control, among other things:

* Weights and activations bit width
* Weights and activation quantization format (int vs float, asym vs sym, etc.)
* PTQ algorithms to apply and their options
* and much more...

We have recently added the possibility to export directly to GGUF after quantization, targetting Q4_0 and Q4_1.

In terms of command line arguments, this corresponds to the following configurations:

.. code-block:: bash

   brevitas_ptq_llm --model org/model --weight-bit-width 4 --weight-quant-type sym --weight-quant-granularity per_group --weight-group-size 32 --export-target gguf:q4_0


.. code-block:: bash

    brevitas_ptq_llm --model org/model --weight-bit-width 4 --weight-quant-type sym --weight-quant-granularity per_group --weight-quant-type asym --weight-group-size 32 --export-target gguf:q4_1

If activation bit-width is not specifies, weight-only quantization will be performed.

These commands will produce quantized models without any extra pre-processing, but several PTQ algorithms are compatible with weight-only quantization.
Among these:

* GPTQ
* AWQ
* Learned Round
* QuaRot/SpinQuant (without any online hadamard rotation)
* MagR

Once the model is exported, it can be used as any other GGUF model.

Direct export and quantized scales/zero_points
==============================================

Although our LLM entrypoint is highly customizable, it still does not expose all the flexibility that Brevitas offers.


In particular, Brevitas allows easily to quantize scale and zero points, matching the new GGUF formats like Q4_K.

This can be done by defining three custom quantizers.
The first two are quantizers that specify how the scales and the zero point must be quantized (i.e., for
the superblocks in the _K GGUF formats). The third one is the base quantizer combining everything together.

This looks like:

.. code-block:: python

    class ShapeMixin(ExtendedInjector):

        @value
        def scaling_shape(
                scaling_per_output_type,
                scaling_per_output_channel_shape,
                expanded_groupwise_shape,
                group_dim):
            if scaling_per_output_type == ScalingPerOutputType.TENSOR:
                scaling = SCALAR_SHAPE
            elif scaling_per_output_type == ScalingPerOutputType.CHANNEL:
                scaling = scaling_per_output_channel_shape
            elif scaling_per_output_type == ScalingPerOutputType.GROUP:
                # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
                assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
                assert group_dim is not None, "Per Group scaling not correctly configured"
                size = list(expanded_groupwise_shape)
                size[group_dim + 1] = 1
                return tuple(size)

            return scaling


    class QuantScalingInt(Int8WeightPerTensorFloat, ShapeMixin):
        bit_width = 6
        module = (this << 1).module

        rescaling_int_quant = RescalingIntQuant
        group_size = 8
        scaling_per_output_type = ScalingPerOutputType.GROUP
        upstream_shape = (this << 1).scaling_shape
        signed = False

        @value
        def tracked_parameter_list(upstream_shape):
            return [torch.empty(upstream_shape)]


    class QuantZPInt(Int8WeightPerTensorFloat, ShapeMixin):
        module = (this << 1).module

        rescaling_int_quant = RescalingIntQuant
        restrict_threshold_impl = FloatRestrictValue
        bit_width = 6
        scaling_per_output_type = ScalingPerOutputType.GROUP
        group_size = 8
        upstream_shape = (this << 1).zero_point_shape
        signed = False

        @value
        def tracked_parameter_list(upstream_shape):
            return [torch.empty(upstream_shape)]


    class QuantScaleQuantZPInt8WeightPerTensorFloat(ShiftedUint8WeightPerTensorFloat):
        proxy_class = GroupwiseWeightQuantProxyFromInjector
        scaling_quant = QuantScalingInt
        zp_quant = QuantZPInt
        restrict_scaling_impl = QuantRestrictValue
        scaling_per_output_type = ScalingPerOutputType.GROUP
        restrict_threshold_impl = FloatRestrictValue
        scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint
        group_size = 32
        bit_width = 4

        @value
        def restrict_value_float_to_int_impl():
            return this.scaling_quant.rescaling_int_quant

        @value
        def zp_int_quant():
            return this.zp_quant.rescaling_int_quant

        @value
        def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
            if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
                return None
            elif scaling_per_output_type == ScalingPerOutputType.GROUP:
                return scaling_shape

        @value
        def zero_point_dequantized_shape(scaling_per_output_type, zero_point_shape):
            if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
                return None
            elif scaling_per_output_type == ScalingPerOutputType.GROUP:
                return zero_point_shape


The intuition behind these quantizers is as follows:
`QuantScaleQuantZeroPointInt8WeightPerTensorFloat` is the baseline quantizer, with asymmetric group-wise quantization at 4 bit.

This quantizer specified two classes used for scale and zero_point quantization:

* restrict_scaling_impl set to QuantRestrictValue, which is responsible for the scale
* scale_shift_zero_point_impl set to _ScaleShiftQuantZeroPoint, responsible for the zero_point

In order to construct these two classes through dependency injection, we need to define `restrict_value_float_to_int_impl` and
`zp_int_quant`, which is done through two `value` functions, a detail of the dependency injection package we use in Brevitas 
(for more info about this, check our `Anatomy of a quantizer tutorial`).

These value functions select the object to instantiate from two other variables defined in the main quantizer, `scaling_quant` and `zp_quant`.

These two variables contain the scale and zero point quantizer, respectively:

* QuantScalingInt
* QuantZeroPointInt

For all practical purposes, these two quantizers behave exactly as any other Brevitas quantizer.
The main exceptions are that they are not directly attached to any layer, but rather to another quantizer.
 
Starting from a standard 8-bit integer quantizer, some parameters are re-defined to match the 
`Q4_K recipe <https://huggingface.co/docs/hub/gguf>`_ , in particular:

* Group-wise quantization 
* Group size equals to 8 (as in, the super block is composed of 8 blocks of 32 elements)
* Bit width is set to 6
* Quantization is unsigned, as we assume that both scales and zero point are defined positive

The other elements are needed for a correct definition of a Brevitas quantizer through dependency injection,
and can be ignored and left as they are.


After they have been created, it is possible to manually create a quant layer as follow:

.. code-block:: python

  qnn.QuantLinear(IN_CHANNEL, OUT_CHANNEL, weight_quant=QuantScaleQuantZeroPointInt8WeightPerTensorFloat)

Alternatively, it is possible to programmatically quantize your network with these quantizers with:

.. code-block:: python

    model = ...
    layer_map[torch.nn.Linear] = (
        qnn.QuantLinear, {
            'weight_quant': QuantScaleQuantZeroPointInt8WeightPerTensorFloat})
    model = layerwise_quantize(
        model=model, compute_layer_map=layer_map)


After quantization is applied, all the same considerations made above for PTQ hold true, and QAT is also a possibility.

Changing the `weight_scaling_impl_type` in the scale and zero_point quantizer to `parameter_from_stats` should
also allow to learn the scale factors of the scale and zero point with QAT, although this is not tested.

After the model is quantized, it is possible to export it with the following:


.. code-block:: python

    from brevitas_examples.llm.gguf_export.export import save_quantized_as_gguf

    save_quantized_as_gguf("/path/to/exported/model", model=model.cpu(), backend="gguf:q4_k_s", tokenizer=tokenizer)


FAQ
===


* *How to export in GGUF format X?*

If you want to quantize and export in GGUF format that is not currently supported, feel free to open an issue.
In general, the indications above, combined with the export code itself, should provide a solid
blueprint to add new export formats, especially similar ones like Q3_K, etc.


* *How to export in Q4_K but still do PTQ?*

We plan to expand the options available in our LLM entrypoint, but introducing scale and zero point quantization
could limit the readability and usability of the script.
If you want to do Q4_K and apply one or more of the algorithms we propose, just follow the same style used in the entrypoint
and write your own quantization script, focusing on one or a few configurations that are of interest for you.


* *Accuracy/Quality of the models seem to be worse compared to other GGUF model. Is it normal?*

Generally speaking, different quantizers have different ways of achieving the same target format.
If the quality you get is not up to your expectations, feel free to try some of the PTQ algorithms suggested above.

If that still does not help, please open an issue and we will be more than happy to look into it.

