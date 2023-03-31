# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import torch

from brevitas import torch_version

if torch_version < version.parse('1.7'):
    from .backport.torch_function import get_testing_overrides
    from .backport.torch_function._overrides import is_tensor_method_or_property
else:
    from torch.overrides import get_testing_overrides
    from torch.overrides import is_tensor_method_or_property

if torch_version < version.parse('1.8.1'):
    from .backport.graph import Graph
    from .backport.graph import magic_methods
    from .backport.graph import reflectable_magic_methods
    from .backport.graph import Target
    from .backport.graph_module import GraphModule
    from .backport.immutable_collections import immutable_dict
    from .backport.immutable_collections import immutable_list
    from .backport.node import map_arg
    from .backport.proxy import base_types
    from .backport.proxy import Node
    from .backport.proxy import Proxy
    from .backport.symbolic_trace import _autowrap_check
    from .backport.symbolic_trace import _find_proxy
    from .backport.symbolic_trace import _orig_module_call
    from .backport.symbolic_trace import _orig_module_getattr
    from .backport.symbolic_trace import _patch_function
    from .backport.symbolic_trace import _Patcher
    from .backport.symbolic_trace import _wrapped_fns_to_patch
    from .backport.symbolic_trace import _wrapped_methods_to_patch
    from .backport.symbolic_trace import HAS_VARSTUFF
    from .backport.symbolic_trace import map_aggregate
    from .backport.symbolic_trace import Tracer
else:
    from torch.fx import Graph
    from torch.fx import GraphModule
    from torch.fx import map_arg
    from torch.fx import Node
    from torch.fx import Proxy
    from torch.fx import Tracer
    from torch.fx.graph import magic_methods
    from torch.fx.graph import reflectable_magic_methods
    from torch.fx.graph import Target
    from torch.fx.proxy import base_types
    try:
        from torch.fx.immutable_collections import immutable_dict
        from torch.fx.immutable_collections import immutable_list
        from torch.fx.symbolic_trace import _autowrap_check
        from torch.fx.symbolic_trace import _find_proxy
        from torch.fx.symbolic_trace import _orig_module_call
        from torch.fx.symbolic_trace import _orig_module_getattr
        from torch.fx.symbolic_trace import _patch_function
        from torch.fx.symbolic_trace import _Patcher
        from torch.fx.symbolic_trace import _wrapped_fns_to_patch
        from torch.fx.symbolic_trace import _wrapped_methods_to_patch
        from torch.fx.symbolic_trace import HAS_VARSTUFF
        from torch.fx.symbolic_trace import map_aggregate
    except ImportError:
        from torch.fx._symbolic_trace import _orig_module_call
        from torch.fx._symbolic_trace import _orig_module_getattr
        from torch.fx._symbolic_trace import _autowrap_check
        from torch.fx._symbolic_trace import _Patcher, map_aggregate
        from torch.fx._symbolic_trace import _wrapped_fns_to_patch, _wrapped_methods_to_patch
        from torch.fx._symbolic_trace import _find_proxy, _patch_function, HAS_VARSTUFF
        from torch.fx.immutable_collections import immutable_dict, immutable_list

from .brevitas_tracer import brevitas_symbolic_trace
from .brevitas_tracer import brevitas_value_trace
from .brevitas_tracer import symbolic_trace
from .brevitas_tracer import value_trace
