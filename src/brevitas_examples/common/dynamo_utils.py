from contextlib import nullcontext
import functools

from packaging import version

from brevitas import torch_version


def patch_dynamo_export():
    # torch 2.10/2.11 crash with "'NoneType' has no attribute 'is_tensor'" when a
    # raw None is left on the dynamo stack (fixed in 2.12 via pytorch#169325).
    # Backport: coerce raw None to ConstantVariable(None) when gathering stack values.
    if not (version.parse('2.10') <= torch_version < version.parse('2.12')):
        return
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.variables import ConstantVariable
    if getattr(OutputGraph._get_stack_values_to_restore, '_brevitas_none_patch', False):
        return
    original_fn = OutputGraph._get_stack_values_to_restore

    @functools.wraps(original_fn)
    def _get_stack_values_to_restore(self, tx, stack_pops):
        stack_values, meta = original_fn(self, tx, stack_pops)
        stack_values = [ConstantVariable.create(None) if v is None else v for v in stack_values]
        return stack_values, meta

    _get_stack_values_to_restore._brevitas_none_patch = True
    OutputGraph._get_stack_values_to_restore = _get_stack_values_to_restore


def dynamo_export_ctx():
    # transformers 5.x checks `arg_name in func.__code__.co_varnames` inside model
    # forward wrappers. torch._dynamo < 2.7 cannot trace __contains__ on a code
    # object descriptor, raising Unsupported. Fail early with a clear message.
    try:
        import transformers as _transformers
        _tr_ver = version.parse(_transformers.__version__)
    except ImportError:
        _tr_ver = version.parse("0")

    if torch_version < version.parse("2.7") and _tr_ver >= version.parse("5.0"):
        raise RuntimeError(
            f"FX-based quantization (dynamo export) is not supported with "
            f"torch < 2.7 and transformers >= 5.0. "
            f"Found torch {torch_version}, transformers {_transformers.__version__}. "
            f"Please upgrade torch to >= 2.7.")

    # torch >= 2.10 inlines nn modules by default; restore call_module behaviour.
    if torch_version >= version.parse('2.10'):
        import torch._dynamo.config as dynamo_config
        return dynamo_config.patch(install_free_tensors_for_export=False)
    return nullcontext()
