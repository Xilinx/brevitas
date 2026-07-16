from contextlib import nullcontext
import functools

from packaging import version

from brevitas import torch_version


def patch_dynamo_export():
    # torch._dynamo.export in torch 2.10 and 2.11 can crash with
    # "'NoneType' object has no attribute 'is_tensor'" when a raw None is left on
    # Dynamo's symbolic stack: OutputGraph.compile_subgraph calls x.is_tensor()
    # over those stack values. Upstream fixed this in 2.12 by replacing raw None
    # on the stack with ConstantVariable(None) (pytorch/pytorch#169325). We
    # backport that behaviour out-of-source for 2.10/2.11 only by coercing raw
    # None to a ConstantVariable when the stack values are gathered.
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
    # Guard: torch < 2.7 combined with transformers >= 5.0 is incompatible for
    # FX-based dynamo export.  In transformers 5.x the @merge_with_config_defaults
    # decorator accesses func.__code__.co_varnames inside the model forward wrapper
    # (``if arg_name in func.__code__.co_varnames``).  torch._dynamo on torch < 2.7
    # cannot trace __contains__ on a GetSetDescriptorVariable (the co_varnames code
    # object descriptor), raising torch._dynamo.exc.Unsupported.  This was fixed in
    # torch 2.7.  Raise a clear RuntimeError early rather than surfacing an obscure
    # dynamo traceback.
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

    # From torch 2.10 onwards, torch._dynamo.export inlines built-in nn modules
    # (install_free_tensors_for_export=True) instead of emitting call_module
    # nodes. Setting install_free_tensors_for_export=False routes them back
    # through the specialized NNModuleVariable path, restoring the pre-2.10 graph
    # structure. The flag does not exist before torch 2.10.
    if torch_version >= version.parse('2.10'):
        import torch._dynamo.config as dynamo_config
        return dynamo_config.patch(install_free_tensors_for_export=False)
    return nullcontext()
