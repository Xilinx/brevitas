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
    # From torch 2.10 onwards, torch._dynamo.export inlines built-in nn modules
    # (install_free_tensors_for_export=True) instead of emitting call_module
    # nodes. Setting install_free_tensors_for_export=False routes them back
    # through the specialized NNModuleVariable path, restoring the pre-2.10 graph
    # structure. The flag does not exist before torch 2.10.
    if torch_version >= version.parse('2.10'):
        import torch._dynamo.config as dynamo_config
        return dynamo_config.patch(install_free_tensors_for_export=False)
    return nullcontext()
