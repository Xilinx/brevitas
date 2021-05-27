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

Forked as-is from PyTorch 1.8.1
"""

import torch
import torch.nn as nn
import linecache
from typing import Type, Dict, List, Any, Union, Optional
from .graph import Graph
import copy
import sys
import traceback
import math
from pathlib import Path
import os
import warnings

# Forked from torch/nn/modules/module.py
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
_next_id = 0
def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)

# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache : Dict[str, List[str]] = {}
_orig_getlines = linecache.getlines
def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)
linecache.getlines = patched_getline

def _forward_from_src(src : str):
    # If you add more globals here, remember to add their names to fx.graph._shadows_builtin_name!
    gbls: Dict[str, Any] = {'inf': math.inf, 'nan': math.nan, 'NoneType' : type(None)}
    exec_with_source(src, gbls)
    return gbls['forward']


def deserialize_graphmodule(body : dict) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """
    # We create a dummy class here because symbolic_trace pulls the forward()
    # function off of the class, rather than the instance
    class CodeOnlyModule(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.__dict__ = body

    try:
        CodeOnlyModule.forward = _forward_from_src(body['_code'])
    except KeyError:
        # BC: attribute name was changed from `code` to `_code` to facilitate
        # making `code` into a property and adding a docstring to it
        CodeOnlyModule.forward = _forward_from_src(body['code'])

    from .symbolic_trace import Tracer

    # we shouldn't trace into any of the submodules, they were not
    # because they were not traced in the original GraphModule
    class KeepModules(Tracer):
        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True

    com = CodeOnlyModule(body)
    return GraphModule(com, KeepModules().trace(com))

# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            # we have already installed one of its parents
            # (e.g. target = root.linear.weight, but we have already installed root.linear)
            # once we install a parent, we no longer need to copy the children
            # since all the needed properties will already be present
            return

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = f, t

    orig = getattr(from_module, field)
    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(from_obj: Any, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    setattr(to_module, field, from_obj)

class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.

    """
    def __new__(cls: 'Type[GraphModule]', *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        class GraphModuleImpl(cls):  # type: ignore
            pass
        return super().__new__(GraphModuleImpl)

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, class_name: str = 'GraphModule'):
        """
        Construct a GraphModule.

        Args:

            root (Union[torch.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

            name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.

        """
        super().__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, torch.nn.Module):
            if hasattr(root, 'training'):
                self.training = root.training
            for node in graph.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    if node.target not in root:
                        raise RuntimeError('Node ' + str(node) + ' referenced target ' + node.target +
                                           ' but that target was not provided in ``root``!')
                    targets_to_copy.append(node.target)
            # Sort targets in ascending order of the # of atoms.
            # This will ensure that less deeply nested attributes are assigned
            # before more deeply nested attributes. For example, foo.bar
            # will be assigned before foo.bar.baz. Otherwise, we might assign
            # the user-provided ``foo.bar`` and wipe out the previously-assigned
            # ``foo.bar.baz``
            targets_to_copy.sort(key=lambda t: t.count('.'))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError('Unsupported type ' + str(root) + ' passed for root!')
        self.graph = graph

    # TorchScript breaks trying to compile the graph setter because of the
    # continued string literal. Issue here: https://github.com/pytorch/pytorch/issues/44842
    #
    # Shouldn't be an issue since these methods shouldn't be used in TorchScript anyway
    __jit_unused_properties__ = ['graph']

    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    @graph.setter
    def graph(self, g) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        self._graph = g
        self.recompile()

    def to_folder(self, folder: Union[str, os.PathLike], module_name : str = "FxModule"):
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
        folder = Path(folder)
        Path(folder).mkdir(exist_ok=True)
        torch.save(self.state_dict(), folder / 'state_dict.pt')
        tab = " " * 4
        model_str = f"""
import torch
from torch.nn import *
class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

        def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
            safe_reprs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules = []
        for module_name, module in self.named_children():
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f'{module_name}.pt'
                torch.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace('\r', ' ').replace('\n', ' ')
                module_str = f"torch.load(r'{module_file}') # {module_repr}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in self._buffers.items():
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}))\n"

        for param_name, param in self._parameters.items():
            model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(buffer.shape)}))\n"

        model_str += f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
        model_str += f"{_addindent(self.code, 4)}\n"

        module_file = folder / 'module.py'
        module_file.write_text(model_str)

        init_file = folder / '__init__.py'
        init_file.write_text('from .module import *')

        if len(blobified_modules) > 0:
            warnings.warn("Was not able to save the following children modules as reprs -"
                          f"saved as pickled files instead: {blobified_modules}")

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, '_code'):
            raise RuntimeError('Code has not been generated! Please report a bug to PyTorch')
        return self._code

    def recompile(self) -> None:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        self._code = self._graph.python_code(root_module='self')
        cls = type(self)
        cls.forward = _forward_from_src(self._code)

        cls_call = cls.__call__

        def print_full_traceback(exctype, value, tb):
            traceback.print_exception(exctype, value, tb)

        def wrapped_call(self, *args, **kwargs):
            old_excepthook = sys.excepthook
            try:
                sys.excepthook = print_full_traceback
                return cls_call(self, *args, **kwargs)
            finally:
                sys.excepthook = old_excepthook
        cls.__call__ = wrapped_call

    def __reduce__(self):
        """
        Serialization of GraphModule. We serialize only the generated code, not
        the underlying ``Graph``. This is because ``Graph`` does not have on-disk
        backward-compatibility guarantees, whereas Python source code does.
        On the deserialization side, we symbolically trace through the generated
        code to regenerate the underlying ``Graph``
        """
        dict_without_graph = self.__dict__.copy()
        del dict_without_graph['_graph']
        return (deserialize_graphmodule, (dict_without_graph,))

    # because __reduce__ is defined for serialization,
    # we need to define deepcopy otherwise it will call __reduce__
    # and cause symbolic tracing to occur every time we try to copy the object
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return GraphModule(fake_mod, self.graph)

    def __copy__(self):
        return GraphModule(self, self.graph)

    def __str__(self) -> str:
        orig_str = super().__str__()
        return '\n'.join([orig_str, self._code])

# workarounds for issues in __torch_function__

# WAR for __torch_function__ not handling tensor lists,
# fix is in https://github.com/pytorch/pytorch/pull/34725
# orig_cat = torch.cat
# def patched_cat(*args, **kwargs):
#     tensors = args[0]
#     for t in tensors:
#         if isinstance(t, Proxy):
#             return t.__torch_function__(patched_cat, (), args, kwargs)
#     return orig_cat(*args, **kwargs)
# patched_cat.__module__ = 'torch'
# patched_cat.__name__ = 'cat'
# torch.cat = patched_cat
