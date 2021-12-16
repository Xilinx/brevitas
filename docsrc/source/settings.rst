========
Settings 
========

Brevitas supports a few boolean global flags can be set through enviromental variables and/or at runtime in the `brevitas.config` package.

`BREVITAS_JIT` (Default: 0) - `brevitas.config.JIT_ENABLED`
Enable just-in-time compilation of built-in quantizers written in TorchScript and of gradient estimators written in C++. 
In case an appropriate C++ compiler cannot be found, a warning will be generated and execution will fall back to the Python implementation.
TorchScript compilation requires PYTORCH_JIT to be enabled (which it is by default).  

`BREVITAS_IGNORE_MISSING_KEYS` (Default: 0) - `brevitas.config.IGNORE_MISSING_KEYS`
Ignore "missing keys" errors typically generated whenever a pretrained floating-point is loaded on top of a corresponding Brevitas 
quantized model that internally contains learned `torch.nn.Parameter` (which happens by default for quantized activations).
This is a better alternative to setting `model.load_state_dict(..., strict=False)`, which would silently ignore any kind of mismatch.

`BREVITAS_VERBOSE` (Default: 0) - `brevitas.config.VERBOSE`
Enable verbose C++ compilation when `BREVITAS_JIT=1` is set.