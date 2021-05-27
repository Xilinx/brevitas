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

from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, Target, map_arg
from .proxy import Proxy
from .symbolic_trace import Tracer
from typing import Any, Dict, Iterator, Optional, Tuple

class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(self, target : Target,
                                  args : Tuple, kwargs : Dict) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : Target,
                                args : Tuple, kwargs : Dict) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_allclose(result, torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The module to be executed
    """
    def __init__(self, module : GraphModule):
        assert isinstance(module, GraphModule)
        self.module = module
        self.submodules = dict(self.module.named_modules())
        self.env : Dict[Node, Any] = {}

    def run(self, *args, initial_env : Optional[Dict[Node, Any]] = None) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter : Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            self.env[node] = self.run_node(node)

            if node.op == 'output':
                output_val = self.env[node]
                return output_val

    def run_node(self, n : Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n.target, args, kwargs)

    # Main Node running APIs

    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)
        if target.startswith('*'):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            return next(self.args_iter)

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        assert isinstance(target, str)
        return self.fetch_attr(target)

    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result
        return target(*args, **kwargs)

    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args  # type: ignore

        # Execute the method and return the result
        assert isinstance(target, str)
        return getattr(self_obj, target)(*args_tail, **kwargs)

    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        return submod(*args, **kwargs)

    def output(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]

    # Helper methods

    def fetch_attr(self, target : str):
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualfiied name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def fetch_args_kwargs_from_env(self, n : Node) -> Tuple[Tuple, Dict]:
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        args = self.map_nodes_to_values(n.args, n)
        assert isinstance(args, tuple)
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        return args, kwargs

    def map_nodes_to_values(self, args : Argument, n : Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """
        def load_arg(n_arg : Node) -> Any:
            if n_arg not in self.env:
                raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                                   f'to diagnose such issues')
            return self.env[n_arg]
        return map_arg(args, load_arg)

class Transformer(Interpreter):
    """
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)

            transformed : torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_allclose(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    """
    def __init__(self, module):
        super().__init__(module)
        self.new_graph = Graph()

        class TransformerTracer(Tracer):
            def __init__(self, graph: Graph):
                super().__init__()
                self.graph = graph

            def is_leaf_module(self, _, __) -> bool:
                return True
        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Proxy:
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        assert isinstance(target, str)
        return Proxy(self.new_graph.placeholder(target), self.tracer)

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Proxy:
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        assert isinstance(target, str)
        return Proxy(self.new_graph.get_attr(target), self.tracer)

    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Override so that the leaf module policy from `self.tracer` is respected.
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        result = super().run()
        if result is not None:
            assert isinstance(result, Proxy)
            self.new_graph.output(result.node)
        return GraphModule(self.module, self.new_graph)
