from packaging import version

import torch

from brevitas import torch_version


if torch_version < version.parse('1.7'):
    from .backport.torch_function._overrides import is_tensor_method_or_property
    from .backport.torch_function import get_testing_overrides
else:
    from torch.overrides import is_tensor_method_or_property
    from torch.overrides import get_testing_overrides

if (torch_version < version.parse('1.8')
        or torch_version >= version.parse('1.9')):
    from .backport.node import map_arg
    from .backport.symbolic_trace import Tracer
    from .backport.graph_module import GraphModule
    from .backport.graph import Target, Graph
    from .backport.proxy import base_types, Proxy, Node
    from .backport.graph import magic_methods, reflectable_magic_methods
    from .backport.symbolic_trace import _orig_module_call
    from .backport.symbolic_trace import _orig_module_getattr
    from .backport.symbolic_trace import _autowrap_check
    from .backport.symbolic_trace import _Patcher, map_aggregate
    from .backport.symbolic_trace import _wrapped_fns_to_patch, _wrapped_methods_to_patch
    from .backport.symbolic_trace import _find_proxy, _patch_function, HAS_VARSTUFF
    from .backport.immutable_collections import immutable_dict, immutable_list
else:
    from torch.fx import map_arg
    from torch.fx import Tracer, Graph, GraphModule, Proxy, Node
    from torch.fx.graph import Target
    from torch.fx.proxy import base_types
    from torch.fx.graph import magic_methods, reflectable_magic_methods
    from torch.fx.symbolic_trace import _orig_module_call
    from torch.fx.symbolic_trace import _orig_module_getattr
    from torch.fx.symbolic_trace import _autowrap_check
    from torch.fx.symbolic_trace import _Patcher, map_aggregate
    from torch.fx.symbolic_trace import _wrapped_fns_to_patch, _wrapped_methods_to_patch
    from torch.fx.symbolic_trace import _find_proxy, _patch_function, HAS_VARSTUFF
    from torch.fx.immutable_collections import immutable_dict, immutable_list

from .brevitas_tracer import value_trace, symbolic_trace
from .brevitas_tracer import brevitas_symbolic_trace, brevitas_value_trace