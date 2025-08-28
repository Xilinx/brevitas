========
Settings
========

Brevitas supports a few boolean global flags can be set through enviromental variables and/or at runtime in the `brevitas.config` package.

`BREVITAS_JIT` (Default: 0) - `brevitas.config.JIT_ENABLED`
Enable just-in-time compilation of built-in quantizers written in TorchScript and of gradient estimators written in C++.
In case an appropriate C++ compiler cannot be found, a warning will be generated and gradient estimators will fall back to the Python implementation.
TorchScript compilation requires PYTORCH_JIT to be enabled (which it is by default).

`BREVITAS_IGNORE_MISSING_KEYS` (Default: 0) - `brevitas.config.IGNORE_MISSING_KEYS`
Ignore "missing keys" errors typically generated whenever a pretrained floating-point is loaded on top of a corresponding Brevitas
quantized model that internally contains learned `torch.nn.Parameter` (which happens by default for quantized activations).
This is a better alternative to setting `model.load_state_dict(..., strict=False)`, which would silently ignore any kind of mismatch.

`BREVITAS_VERBOSE` (Default: 0) - `brevitas.config.VERBOSE`
Enable verbose C++ compilation when `BREVITAS_JIT=1` is set.

`BREVITAS_NATIVE_STE_BACKEND` (Default: 0) - `brevitas.config.NATIVE_STE_BACKEND_ENABLED`
Enable compilation of the native C++ backend without enabling the JIT for Brevitas.
Setting `brevitas.config.JIT_ENABLED=True` forces `brevitas.config.NATIVE_STE_BACKEND_ENABLED=True`.

`BREVITAS_REINIT_ON_STATE_DICT_LOAD` (Default: 1) - `brevitas.config.REINIT_ON_STATE_DICT_LOAD`
Controls whether loading a state dict triggers re-initialization of the quantizers within the layer whose state dict is being updated.
The default behaviour is for quantizers to be re-initialized, so that any quantization parameter that is initialized based on floating-point
parameters can be recomputed, e.g. a learned `torch.nn.Parameter` weight scale initialized from the absmax of the weight float tensor.
Disable at your own risk if it's appropriate for your use case.

`BREVITAS_LOGGING` (Default: CRITICAL) - `brevitas.config.BREVITAS_LOGGING`
Controls how much feedback the various operations within Brevitas will provide to the user.
From most to least verbose the current available options are: DEBUG, INFO, CRITICAL.
