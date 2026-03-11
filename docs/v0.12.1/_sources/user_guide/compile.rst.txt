====================
Brevitas and Compile
====================


Flexibility and Fake quantization
=================================
Brevitas is inherently based around the concepts of flexibility and fake quantization,
The first allows us to combine and test lots of different options for quantization,
for example computing scale factors with AbsMax statistics and zero point with MSE, or compute both of them with percentile,
but picking different percentile values for each of them.
The combinations are endless.
Similarly, this flexibility allows us to have shared infrastructure for PTQ and QAT,
which from our point of view (the maintainers), it means less code to maintain and test,
and from the users perspective, it opens new possibilities like a smooth transition from PTQ to QAT with all the related benefits of that.
The idea of fake quantization instead allows us to replicate and mimic a wide combination of data-types even when the underlying hardware does not support them.

Both of these concepts come with some caveats,
in particular both flexibility and fake quantization are inherently difficult to optimize and,
in eager mode, computationally slower compared to a non-quantized network.

Flexibility means we cannot optimize and fuse together operations with specialized kernels,
unless we decide to have specific optimizations for the "most common" configurations.
Similarly, fake quantization means we do not rely on fast kernel with reduced precision data-types;
this could be because the data-type we are representing is not supported in hardware, or even if it is,
there could be no optimized kernels that accounts for quantization in the loop.

This is well-known price to pay for quantization, with the idea that at inference time,
once all the decisions have been made and flexibility is not a concern anymore,
it is possible to deploy a quantized network with highly optimized kernels and great speed-ups compared to their floating point variant.



Compile your quantized network
==============================
Starting with torch 2.0, it is now possible to `compile` your code to get on-the-fly speed-up compared to eager execution.
With the most recent versions of torch, this functionality is greatly improved with support to more operations and patterns.

However, compile still has some limitations, which might lead to excessive recompilations or failures to compile altogether.

In Brevitas, we are adding support for compile in different point of the quantization pipeline,
trying to find a good compromise between ease-of-use, speed-up, and compatibility.

Currently, there are three main ways to leverage ``torch.compile`` with Brevitas, each with its own pros and cons.

The first two of these approaches rely on newly introduced ``quant_inference_mode``.
This mode should be used once quantization is finished, and the idea is to trade away some flexibility,
which is not needed anymore at inference time, in exchange for slightly faster execution times.

Full model compile + ``quant_inference_mode``
---------------------------------------------

The first option is to compile the entire model after entering ``quant_inference_mode``.
Using quant_inference_mode simplifies the compute graph that compile needs to optimize,
as well as drop the use of ``QuantTensor``.
The NamedTuple structure is not currently compatible with compile, and even the torch subclass tensors have some outstanding issues.

This approach might grant the best performance, since the entire model is being optimized.
On the negative side, model-specific issues might cause compile to fail.
Similarly, it is easier to fall into too-many-recompilations issue, and the compile process might be extremely slow,
which means that it becomes beneficial only for long inference runs.
In this scenario, the user is responsible to compile the model, as in the example below:

.. code-block:: python
    
    model = MyQuantModel()
    # Quantization code goes here
    ...
    # Inference
    with torch.no_grad(), with quant_inference_mode(model):
        # We need one forward pass to cache quantization-related hyper-params
        model(**example_input)
        model = torch.compile(model)


Quantizers compile + ``quant_inference_mode``
---------------------------------------------
The second approach tied to ``quant_inference_mode`` is the compilation of the quantization functions.
We noticed that this is already enough to grant a considerable speed-up (numbers below) for some use cases,
and the lower surface area means that we can control a bit better any potential ``torch.compile`` issue.
Also, the compilation time is greatly reduced compared to the previous case,
although the speed-up benefits will also be slightly lower.


.. code-block:: python

    model = MyQuantModel()
    # Quantization code goes here
    ...
    # Inference
    with torch.no_grad(), with quant_inference_mode(model, compile=True):
        # We need one forward pass to cache quantization-related hyper-params
        model(**example_input)

Compared to the previous case, compile is handled automatically by the context manager.

Quantizers compile + PTQ
------------------------

The third option is to compile the quantization functions without ``quant_inference_mode``.
In this case, the computational graph is slightly more complicated,
and the possibility of errors with torch.compile increases.
The benefit of this approach is that it can be used also during PTQ,
so not only inference time, which for some algorithms is definitely interesting.

.. code-block:: python

    model = MyQuantModel()
    for m in model.modules():
        if hasattr(m, 'compile_quant'):
            m.compile_quant()
    
    # Quantization code goes here
    ...
    # Inference
    with torch.no_grad(), with quant_inference_mode(model, compile=True):
        # We need one forward pass to cache quantization-related hyper-params
        model(**example_input)

As in the previous case, the user is responsible for compiling the model,
although we provide some functions in our quantizers to simplify the process.
NB: this interface might (and very likely will) change in the future.
This approach is also compatible with ``quant_inference_mode``, although it requires to reset
compilation status, which is not thoroughly tested.


Some results
============

Quantizers compile + ``quant_inference_mode``
---------------------------------------------
These are small examples of possible speed-ups with compile.
The runtime includes compilation time, which is especially significant for the WikiText2 inference that has a very short runtime.
Even then, compile provides a considerable speed-up,
which becomes more evident with bigger models and longer evaluations (e.g., few-shot).



.. list-table:: Sana 1.6B, with per-group fp8 quantization
   :widths: 25 25 25
   :header-rows: 1

   * - Quant Type
     - Compile Inference Time (500 samples)
     - Eager Inference Time (500 samples)
   * - Float
     - Not Measured
     - 25m
   * - Weight-only quantization
     - 26m
     - 1h14m
   * - Act + Weight quantization
     - 1h15m
     - 2h10m


.. list-table:: Llama 3.2 1B, with per-group fp8 quantization
   :widths: 25 25 25
   :header-rows: 1

   * - Quant Type
     - Compile Inference Time (WikiText2)
     - Eager Inference Time (WikiText2)
   * - Float
     - Not Measured
     - 12s
   * - Weight-only quantization
     - 18s
     - 40s
   * - Act + Weight quantization
     - 40s
     - 1m

Known Gotchas
=============

Although lots of steps were taken to make Brevitas as compile-friendly as possible,
there are some known cases where recompilations are still necessary or errors might arise.
A non-comprehensive list can be found below:

* Dynamic Activation quantization requires recompilations, even within inference mode

* Compiling the entire model after optimizing for PTQ requires resetting the compilation status (e.g., ``torch._dynamo.reset()``)

* Some operations are currently not supported for compile, such as kth-value that we use for percentile statistics

* When optimizing PTQ, it is generally suggested to skip the activation calibration part, as it may lead to too-many-recompilations errors

* Compiling inference execution might lead to slightly different output compared to eager execution

* Compiling PTQ and inference might lead to a more marked difference in outputs compared to eager execution

* Although we investigated some use cases when compiling quantizers, we did not test all possible combinations

* We definitely tested very few compile + PTQ cases

* Some torch versions might have compile regressions


FAQ
===

For all the questions below, opening an issue to seek further clarifications is always an option and it is encouraged.
Please provide minimal example so that we can reproduce your issue.


* *Compiling the entire model in quant_inference_mode fails, can you help?*

First it is important to understand whether the error is due to the model itself or quantization.
Even if compilation fails only with quantization in the loop, it might be too broad for us to fix without over-specialization of code.


* *Combining quant_inference_mode with compile=True gives me too-many-recompilations error, what should I do?*

Increasing ``torch._dynamo.config.cache_size_limit`` or ``torch._dynamo.config.accumulated_cache_size_limit`` might help. 

* *After compiling, I don't see any speed-up. Is this normal?*

Yes, for some combinations of quantizers, compile might provide limited benefits.
We noticed that minifloat quantization benefits more than integer one, especially with ``quant_inference_mode``.
Similarly, compiling during PTQ might not provide benefits because the slow part of the codebase is not the quantization part,
but the algorithm itself.

* *Which PTQ algorithms are compile-friendly?*

This is undefined.
In general, it does not only depend on the algorithm itself but also on everything that comes after the compilation process.
A lot of supported algorithms should be fairly compatible with compile since there's limited interaction,
but we have not tested all possible combinations with all possible networks.

* *What versions of PyTorch should I use?*

Possibly, always the latest available.
We are trying to ramp-up our tests across PyTorch versions,
but there are a lot of new functionalities and bug-fixes every new versions.

* *I am getting different accuracy with/without compile. Can you fix it?*

No, this is known issue, due to underlying optimizations we cannot control.

* *What are the next steps for Brevitas + compile?*

We would like to expand the optimization area, balancing code refactoring for compile with observed speed-ups.
An example of this is to compile an entire ``QuantLayer``, but we also need to study on the trade-offs.
We would love to increase of test suite for this, and we welcome all contributions.

