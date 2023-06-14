"""
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Forked as-is from PyTorch 2.0.1
"""

from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type

from torch.utils._pytree import LeafSpec
from torch.utils._pytree import PyTree
from torch.utils._pytree import TreeSpec

FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]

SUPPORTED_NODES: Dict[Type[Any], Any] = {}


def register_pytree_flatten_spec(typ: Any, flatten_fn_spec: FlattenFuncSpec) -> None:
    SUPPORTED_NODES[typ] = flatten_fn_spec


def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(
            f"{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with"
            "torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make"
            "sure that any custom pytrees have been registered before loading it.")
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec)
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec)
        result += flat
    return result


def _dict_flatten_spec(d: Dict[Any, Any], spec: TreeSpec) -> List[Any]:
    return [d[k] for k in spec.context]


def _list_flatten_spec(d: List[Any], spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]


def _tuple_flatten_spec(d: Tuple[Any], spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]


def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]


register_pytree_flatten_spec(dict, _dict_flatten_spec)
register_pytree_flatten_spec(list, _list_flatten_spec)
register_pytree_flatten_spec(tuple, _tuple_flatten_spec)
register_pytree_flatten_spec(namedtuple, _tuple_flatten_spec)
