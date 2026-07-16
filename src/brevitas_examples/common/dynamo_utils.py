from contextlib import contextmanager
import functools

from packaging import version

from brevitas import torch_version


def _patch_transformers_co_varnames():
    """
    Patch transformers 5.x's `merge_with_config_defaults` decorator so its
    wrapper does not access ``func.__code__.co_varnames`` at call time.

    In transformers 5.x a ``@merge_with_config_defaults`` wrapper checks
    ``if arg_name in func.__code__.co_varnames:`` on every call.  On torch
    ≤ 2.6, torch._dynamo cannot trace the ``__contains__`` operation on a
    ``GetSetDescriptorVariable`` (the co_varnames descriptor), which causes
    ``torch._dynamo.exc.Unsupported`` for every FX-path test.  The fix is to
    precompute the ``arg_index`` mapping *once* at decoration time (outside the
    wrapper) so the wrapper only operates on plain Python values that dynamo
    can handle.

    Returns True if the patch was applied, False if it was not needed.
    """
    try:
        import transformers.utils.generic as tug
    except ImportError:
        return False

    if not hasattr(tug, 'merge_with_config_defaults'):
        return False

    if getattr(tug.merge_with_config_defaults, '_brevitas_co_varnames_patch', False):
        return False

    original = tug.merge_with_config_defaults

    _ARGS_WITH_CONFIG_DEFAULTS = [
        "use_cache",
        "vision_feature_layer",
        "vision_feature_select_strategy",
        "vision_aspect_ratio",]

    def _patched_merge_with_config_defaults(func):
        # Precompute arg_index for each argument at decoration time so the
        # wrapper never touches func.__code__.co_varnames symbolically.
        co_varnames = func.__code__.co_varnames
        arg_index_map = {}
        for arg_name in _ARGS_WITH_CONFIG_DEFAULTS:
            if arg_name in co_varnames:
                arg_index_map[arg_name] = co_varnames.index(arg_name) - 1  # -1 for self
            else:
                arg_index_map[arg_name] = None

        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for arg_name in _ARGS_WITH_CONFIG_DEFAULTS:
                arg_index = arg_index_map[arg_name]

                if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:
                    arg_value = args[arg_index]
                elif kwargs.get(arg_name) is not None:
                    arg_value = kwargs[arg_name]
                else:
                    arg_value = getattr(self.config, arg_name, None)

                if arg_value is not None:
                    if arg_name == "use_cache":
                        if getattr(self, "gradient_checkpointing",
                                   False) and self.training and arg_value:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(
                                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                            )
                            arg_value = False
                    elif arg_name == "vision_feature_select_strategy":
                        valid_strategies = ["default", "full"]
                        if arg_value not in valid_strategies:
                            raise ValueError(
                                f"`Unexpected select feature strategy: {arg_value}. Please select from {valid_strategies}."
                            )

                    if arg_index is not None and len(args) > arg_index:
                        args = args[:arg_index] + (arg_value,) + args[arg_index + 1:]
                    else:
                        kwargs[arg_name] = arg_value

            return func(self, *args, **kwargs)

        return wrapper

    _patched_merge_with_config_defaults._brevitas_co_varnames_patch = True
    tug.merge_with_config_defaults = _patched_merge_with_config_defaults

    # Re-apply the patched decorator to any already-decorated forward methods
    # that are currently in memory.  This is done best-effort: we iterate over
    # all transformers model classes that are already imported and unwrap/rewrap
    # any forward that was wrapped by the original decorator.
    try:
        import transformers.modeling_utils as tmu
        PreTrainedModel = tmu.PreTrainedModel

        def _get_original(fn):
            """Peel off one layer of functools.wraps."""
            return getattr(fn, '__wrapped__', fn)

        for cls in PreTrainedModel.__subclasses__():
            for name in ('forward', 'model'):
                if not hasattr(cls, name):
                    continue
                method = cls.__dict__.get(name)
                if method is None:
                    continue
                # Check if this looks like a merge_with_config_defaults wrapper:
                # it has __wrapped__ pointing to a function with co_varnames
                inner = _get_original(method)
                if (inner is not method and hasattr(inner, '__code__') and
                        any(a in inner.__code__.co_varnames for a in _ARGS_WITH_CONFIG_DEFAULTS)):
                    setattr(cls, name, _patched_merge_with_config_defaults(inner))
    except Exception:
        pass  # best-effort

    return True


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


@contextmanager
def dynamo_export_ctx():
    # From torch 2.10 onwards, torch._dynamo.export inlines built-in nn modules
    # (install_free_tensors_for_export=True) instead of emitting call_module
    # nodes. Setting install_free_tensors_for_export=False routes them back
    # through the specialized NNModuleVariable path, restoring the pre-2.10 graph
    # structure. The flag does not exist before torch 2.10.
    #
    # Additionally, on torch < 2.7 (specifically 2.6), transformers 5.x's
    # merge_with_config_defaults decorator accesses func.__code__.co_varnames
    # at call-time, which torch.dynamo cannot trace symbolically. We patch
    # this to precompute the indices at decoration time instead.
    _patch_transformers_co_varnames()

    if torch_version >= version.parse('2.10'):
        import torch._dynamo.config as dynamo_config
        with dynamo_config.patch(install_free_tensors_for_export=False):
            yield
    else:
        yield
