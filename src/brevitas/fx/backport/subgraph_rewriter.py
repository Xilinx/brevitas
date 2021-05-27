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
from .node import Node
from .symbolic_trace import symbolic_trace

import copy
from typing import Callable, Dict, List, NamedTuple, Set

class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]

class SubgraphMatcher:
    def __init__(self, pattern : Graph) -> None:
        self.pattern = pattern
        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an "
                             "empty pattern")
        # `self.pattern_anchor` is the output Node in `pattern`
        self.pattern_anchor = next(iter(reversed(pattern.nodes)))
        # Ensure that there is only a single output value in the pattern
        # since we don't support multiple outputs
        assert len(self.pattern_anchor.all_input_nodes) == 1, \
            "Pattern matching on multiple outputs is not supported"
        # Maps nodes in the pattern subgraph to nodes in the larger graph
        self.nodes_map: Dict[Node, Node] = {}

    def matches_subgraph_from_anchor(self, anchor : Node) -> bool:
        """
        Checks if the whole pattern can be matched starting from
        ``anchor`` in the larger graph.

        Pattern matching is done by recursively comparing the pattern
        node's use-def relationships against the graph node's.
        """
        self.nodes_map = {}
        return self._match_nodes(self.pattern_anchor, anchor)

    # Compare the pattern node `pn` against the graph node `gn`
    def _match_nodes(self, pn : Node, gn : Node) -> bool:

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in self.nodes_map:
            return self.nodes_map[pn] == gn

        def attributes_are_equal(pn : Node, gn : Node) -> bool:
            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if (pn.op == "placeholder"
                    or (pn.op == "output" and gn.op != "placeholder")):
                return True
            return pn.op == gn.op and pn.target == gn.target

        # Terminate early if the node attributes are not equal
        if not attributes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        self.nodes_map[pn] = gn

        # Traverse the use-def relationships to ensure that `pn` is a true
        # match for `gn`
        if (pn.op != "output"
                and len(pn.all_input_nodes) != len(gn.all_input_nodes)):
            return False
        if pn.op == "output":
            match_found = any(self._match_nodes(pn.all_input_nodes[0], gn_)
                              for gn_ in gn.all_input_nodes)
        else:
            match_found = (len(pn.all_input_nodes) == len(gn.all_input_nodes)
                           and all(self._match_nodes(pn_, gn_) for pn_, gn_
                                   in zip(pn.all_input_nodes, gn.all_input_nodes)))
        if not match_found:
            self.nodes_map.pop(pn)
            return False

        return True


def replace_pattern(gm : GraphModule, pattern : Callable, replacement : Callable) -> List[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2

    """
    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph = gm.graph
    pattern_graph = symbolic_trace(pattern).graph
    replacement_graph = symbolic_trace(replacement).graph

    # Find all possible pattern matches in original_graph. Note that
    # pattern matches may overlap with each other.
    matcher = SubgraphMatcher(pattern_graph)
    matches: List[Match] = []

    # Consider each node as an "anchor" (deepest matching graph node)
    for anchor in original_graph.nodes:

        if matcher.matches_subgraph_from_anchor(anchor):

            def pattern_is_contained(nodes_map : Dict[Node, Node]) -> bool:
                # `lookup` represents all the nodes in `original_graph`
                # that are part of `pattern`
                lookup: Dict[Node, Node] = {v : k for k, v
                                            in nodes_map.items()}
                for n in lookup.keys():

                    # Nodes that can "leak"...

                    # Placeholders (by definition)
                    if n.op == "placeholder":
                        continue
                    # Pattern output (acts as a container)
                    if lookup[n].op == "output":
                        continue
                    # Result contained by pattern output (what we'll
                    # hook in to the new Graph, thus what we'll
                    # potentially use in other areas of the Graph as
                    # an input Node)
                    if (len(lookup[n].users) == 1
                            and list(lookup[n].users.keys())[0].op == "output"):
                        continue

                    for user in n.users:
                        # If this node has users that were not in
                        # `lookup`, then it must leak out of the
                        # pattern subgraph
                        if user not in lookup:
                            return False
                return True

            # It's not a match if the pattern leaks out into the rest
            # of the graph
            if pattern_is_contained(matcher.nodes_map):
                for k, v in matcher.nodes_map.items():
                    # Shallow copy nodes_map
                    matches.append(Match(anchor=anchor,
                                   nodes_map=copy.copy(matcher.nodes_map)))

    # The set of all nodes in `original_graph` that we've seen thus far
    # as part of a pattern match
    replaced_nodes: Set[Node] = set()

    # Return True if one of the nodes in the current match has already
    # been used as part of another match
    def overlaps_with_prev_match(match : Match) -> bool:
        for n in match.nodes_map.values():
            if n in replaced_nodes and n.op != "placeholder":
                return True
        return False

    for match in matches:

        # Skip overlapping matches
        if overlaps_with_prev_match(match):
            continue

        # Map replacement graph nodes to their copy in `original_graph`
        val_map: Dict[Node, Node] = {}

        pattern_placeholders = [n for n in pattern_graph.nodes
                                if n.op == "placeholder"]
        assert len(pattern_placeholders)
        replacement_placeholders = [n for n in replacement_graph.nodes
                                    if n.op == "placeholder"]
        assert len(pattern_placeholders) == len(replacement_placeholders)
        placeholder_map = {r : p for r, p
                           in zip(replacement_placeholders, pattern_placeholders)}

        # node from `original_graph` that matched with the output node
        # in `pattern`
        subgraph_output: Node = match.anchor

        def mark_node_as_replaced(n : Node) -> None:
            if n not in match.nodes_map.values():
                return
            for n_ in n.all_input_nodes:
                mark_node_as_replaced(n_)
            replaced_nodes.add(n)

        mark_node_as_replaced(subgraph_output)

        # Intialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        for replacement_node in replacement_placeholders:
            # Get the `original_graph` placeholder node
            # corresponding to the current `replacement_node`
            pattern_node = placeholder_map[replacement_node]
            original_graph_node = match.nodes_map[pattern_node]
            # Populate `val_map`
            val_map[replacement_node] = original_graph_node

        # Copy the replacement graph over
        with original_graph.inserting_before(subgraph_output):
            copied_output = original_graph.graph_copy(replacement_graph,
                                                      val_map)

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location

        # CASE 1: We need to hook the replacement subgraph in somewhere
        # in the middle of the graph. We replace the Node in the
        # original graph that corresponds to the end of the pattern
        # subgraph
        if subgraph_output.op != "output":
            # `subgraph_output` may have multiple args. These args could
            # be from the orignal graph, or they could have come from
            # the insertion of `replacement_subgraph`. We need to find
            # the Node that was originally matched as part of
            # `pattern` (i.e. a Node from the original graph). We can
            # figure this out by looking in `match.nodes_map`. The map
            # was created before `replacement_subgraph` was spliced in,
            # so we know that, if a Node is in `match.nodes_map.values`,
            # it must have come from the original graph
            for n in subgraph_output.all_input_nodes:
                if (n.op != "placeholder"
                        and n in match.nodes_map.values()):
                    subgraph_output = n
                    break
            assert subgraph_output.op != "output"
        # CASE 2: The pattern subgraph match extends to the end of the
        # original graph, so we need to change the current graph's
        # output Node to reflect the insertion of the replacement graph.
        # We'll keep the current output Node, but update its args and
        # `_input_nodes` as necessary
        else:
            subgraph_output.args = ((copied_output,))
            if isinstance(copied_output, Node):
                subgraph_output._input_nodes = {copied_output: None}

        assert isinstance(copied_output, Node)
        subgraph_output.replace_all_uses_with(copied_output)

        # Erase the `pattern` nodes
        for node in reversed(original_graph.nodes):
            if len(node.users) == 0 and node.op != "output":
                original_graph.erase_node(node)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    return matches
