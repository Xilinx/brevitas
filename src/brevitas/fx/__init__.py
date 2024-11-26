# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import torch

from brevitas import torch_version

# This needs to be bumped until https://github.com/pytorch/pytorch/pull/94461 gets merged
if torch_version < version.parse('2.5'):
    import brevitas.backport.fx as fx
    from brevitas.backport.fx._compatibility import compatibility
    from brevitas.backport.fx._symbolic_trace import _assert_is_none
    from brevitas.backport.fx._symbolic_trace import _autowrap_check
    from brevitas.backport.fx._symbolic_trace import _find_proxy
    from brevitas.backport.fx._symbolic_trace import _is_fx_tracing_flag
    from brevitas.backport.fx._symbolic_trace import _orig_module_call
    from brevitas.backport.fx._symbolic_trace import _orig_module_getattr
    from brevitas.backport.fx._symbolic_trace import _patch_function
    from brevitas.backport.fx._symbolic_trace import _Patcher
    from brevitas.backport.fx._symbolic_trace import _proxyable_classes
    from brevitas.backport.fx._symbolic_trace import _wrapped_fns_to_patch
    from brevitas.backport.fx._symbolic_trace import _wrapped_methods_to_patch
    from brevitas.backport.fx._symbolic_trace import HAS_VARSTUFF
    from brevitas.backport.fx._symbolic_trace import map_aggregate
    from brevitas.backport.fx._symbolic_trace import PH
    from brevitas.backport.fx._symbolic_trace import Tracer
    from brevitas.backport.fx.graph import _PyTreeCodeGen
    from brevitas.backport.fx.graph import _PyTreeInfo
    from brevitas.backport.fx.graph import Graph
    from brevitas.backport.fx.graph import magic_methods
    from brevitas.backport.fx.graph import reflectable_magic_methods
    from brevitas.backport.fx.graph import Target
    from brevitas.backport.fx.graph_module import GraphModule
    from brevitas.backport.fx.immutable_collections import immutable_dict
    from brevitas.backport.fx.immutable_collections import immutable_list
    from brevitas.backport.fx.node import map_arg
    from brevitas.backport.fx.proxy import base_types
    from brevitas.backport.fx.proxy import Node
    from brevitas.backport.fx.proxy import ParameterProxy
    from brevitas.backport.fx.proxy import Proxy
    from brevitas.backport.fx.proxy import Scope
    from brevitas.backport.fx.proxy import ScopeContextManager
    import brevitas.backport.fx.traceback as fx_traceback
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
    from torch.fx._symbolic_trace import _orig_module_call
    from torch.fx._symbolic_trace import _orig_module_getattr
    from torch.fx._symbolic_trace import _autowrap_check
    from torch.fx._symbolic_trace import _Patcher, map_aggregate
    from torch.fx._symbolic_trace import _wrapped_fns_to_patch, _wrapped_methods_to_patch
    from torch.fx._symbolic_trace import _find_proxy, _patch_function, HAS_VARSTUFF
    from torch.fx.immutable_collections import immutable_dict, immutable_list
    import torch.fx as fx
    from torch.fx._compatibility import compatibility
    from torch.fx._symbolic_trace import _assert_is_none
    from torch.fx._symbolic_trace import PH
    from torch.fx.graph import _PyTreeCodeGen
    from torch.fx.graph import _PyTreeInfo
    from torch.fx.proxy import ParameterProxy
    from torch.fx.proxy import Scope
    from torch.fx.proxy import ScopeContextManager
    import torch.fx.traceback as fx_traceback
    from torch.fx._symbolic_trace import _is_fx_tracing_flag
    from torch.fx._symbolic_trace import _proxyable_classes

from .brevitas_tracer import brevitas_symbolic_trace
from .brevitas_tracer import brevitas_value_trace
from .brevitas_tracer import symbolic_trace
from .brevitas_tracer import value_trace
